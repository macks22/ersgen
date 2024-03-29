import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import preprocessing


# Error codes
BOUNDS_FORMAT = 1000
MISSING_ATTRIBUTE = 1001
DIM_MISMATCH = 1002
BAD_FEATURE_CONF = 1003
BAD_FILENAME = 1004
BAD_CMDLINE_ARG = 1005


class BadFeatureConfig(Exception):
    """Raise when bad feature configuration file is found."""
    pass


def read_feature_guide(fname):
    """Read the feature guide and parse out the specification.

    The expected file format is the following:

        t:<target>;
        e:<comma-separated categorical entity names>;
        c:<comma-separated categorical variable names>;
        r:<comma-separated real-valued variable names>;

    Whitespace is ignored, as are lines that start with a "#" symbol. Any
    variables not included in one of the three groups is ignored. We assume the
    first two categorical variables are the user and item ids.

    Args:
        fname (str): Path of the file containing the feature guide.

    Returns:
        iterable: (target, entity list, categoricals list, real-valued list).

    """
    with open(fname) as f:
        lines = [l.strip() for l in f.read().split('\n')
                 if not l.startswith('#') and l.strip()]

    parsing = False
    keys = ['t', 'e', 'c', 'r']
    vars = {var: [] for var in keys}
    for line in lines:
        if not parsing:
            k, csv = line.split(':')
        else:
            csv = line

        vars[k].extend([val.strip() for val in csv.split(',')])
        parsing = not line.endswith(';')
        if not parsing:
            vars[k][-1] = vars[k][-1][:-1]

    # Remove whitespace strings. These may have come from something like:
    # c: this, , that;
    for k in keys:
        vars[k] = [val for val in vars[k] if val]  # already stripped

    # Sanity checks.
    num_targets = len(vars['t'])
    if num_targets != 1:
        raise BadFeatureConfig(
            'feature conf should specify 1 target; got %d; check for'
            'the semi-colon at the end of the t:<target> line' % num_targets)

    num_bias = len(vars['e'])
    if not num_bias:
        raise BadFeatureConfig('no entity variables given; need at least 1')

    num_features = len(vars['c']) + len(vars['r'])
    if not num_features > 0:
        raise BadFeatureConfig('no predictors specified')

    spec = (vars['t'][0], vars['e'], vars['c'], vars['r'])
    target, ents, cats, reals = spec
    logging.info('read the following feature guide:')
    logging.info('target: %s' % target)
    logging.info('entities: %s' % ', '.join(ents))
    logging.info('categoricals: %s' % ', '.join(cats))
    logging.info('real-valueds: %s' % ', '.join(reals))
    return spec


def map_ids(data, key, id_map=None):
    """Map ids to 0-contiguous index. This enables the use of these ids as
    indices into an array (for the bias terms, for instance). This returns the
    number of unique IDs for `key`.
    """
    if id_map is None:
        ids = data[key].unique()
        n = len(ids)
        id_map = dict(zip(ids, range(n)))
    else:
        data[key] = data[key].apply(lambda _id: id_map[_id])
    return id_map


def read_data(train_file, test_file, conf_file):
    """Read the train and test data according to the feature guide in the
    configuration file. Return the train and test data as (X, y) pairs for the
    train and test data.

    Args:
        train_file (str): Path of the CSV train data file.
        test_file (str): Path of the CSV test data file.
        conf_file (str): Path of the configuration file.

    Returns:
        tuple: (train_eids, train_X, train_y,
                test_eids, test_X, test_y,
                feat_indices, number_of_entities),
               preprocessed train/test data and mappings from features to
               indices in the resulting data matrices. Both train and test are
               accompanied by 0-contiguous primary entity id arrays.

    """
    try:
        target, ents, cats, reals = read_feature_guide(conf_file)
    except IOError:
        raise IOError('invalid feature guide conf file path: %s' % conf_file)

    to_read = [target] + ents + cats + reals
    def read_file(name, fpath):
        try:
            data = pd.read_csv(fpath, usecols=to_read)
        except IOError:
            raise IOError('invalid %s file path: %s' % (name, fpath))
        except ValueError as err:
            attr_name = err.args[0].split("'")[1]
            attr_type = ('entity' if attr_name in ents else
                         'categorical' if attr_name in cats else
                         'real-valued' if attr_name in reals else
                         'target')
            raise BadFeatureConfig('%s attribute %s not found in %s file' % (
                attr_type, attr_name, name))

        return data

    train = read_file('train', train_file)
    test = read_file('test', test_file)
    nd_train = train.shape[0]
    nd_test = test.shape[0]
    logging.info('number of dyads: train=%d, test=%d' % (nd_train, nd_test))

    # Separate X, y for train/test data.
    train_y = train[target].values
    test_y = test[target].values

    # Read out id lists for primary entity.
    id_map = map_ids(train, ents[0])
    map_ids(test, ents[0], id_map)
    train_eids = train[ents[0]].values
    test_eids = test[ents[0]].values

    # Z-score scaling of real-valued features.
    scaler = preprocessing.StandardScaler()
    train_reals = scaler.fit_transform(train[reals])
    test_reals = scaler.transform(test[reals])

    # One-hot encoding of entity and categorical features.
    catf = ents + cats
    all_cats = pd.concat((train[catf], test[catf]))
    encoder = preprocessing.OneHotEncoder()
    enc = encoder.fit_transform(all_cats)
    train_cats = enc[:nd_train]
    test_cats = enc[nd_train:]

    # Create a feature map for decoding one-hot encoding.
    ncats = encoder.active_features_.shape[0]
    nreal = train_reals.shape[1]
    nf = ncats + nreal

    # Count entities.
    logging.info('after one-hot encoding, found # unique values:')
    counts = np.array([
        all_cats[catf[i]].unique().shape[0]
        for i in xrange(len(catf))
    ])
    indices = zip(catf, np.cumsum(counts))
    for attr, n_values in zip(ents, counts):
        logging.info('%s: %d' % (attr, n_values))

    # Add in real-valued feature indices.
    indices += zip(reals, range(indices[-1][1] + 1, nf + 1))

    # How many entity features and categorical features do we have?
    nents = dict(indices)[ents[-1]]
    ncats = ncats - nents
    nf = nents + ncats + nreal

    ent_idx = range(len(ents))
    cat_idx = range(len(ents), len(ents) + len(cats))
    nactive_ents = sum(encoder.n_values_[i] for i in ent_idx)
    nactive_cats = sum(encoder.n_values_[i] for i in cat_idx)

    logging.info('number of active entity features: %d of %d' % (
        nents, nactive_ents))
    logging.info('number of active categorical features: %d of %d' % (
        ncats, nactive_cats))
    logging.info('number of real-valued features: %d' % nreal)

    # Put all features together.
    train_X = sp.sparse.hstack((train_cats, train_reals))
    test_X = sp.sparse.hstack((test_cats, test_reals))
    logging.info('Total of %d features after encoding' % nf)

    return (train_eids, train_X, train_y,
            test_eids, test_X, test_y,
            indices, nents)


def ipr_predict(model, eids, X, nb):
    """Make predictions for each feature vector in X using the IPR model."""
    w0 = model['w0']
    w = model['w']
    P = model['P'][eids]
    W = model['W']

    B = X[:, :nb]
    X = X[:, nb:]
    return w0 + B.dot(w) + (X.dot(W.T) * P).sum(axis=1)


def rmse_from_err(err):
    return np.sqrt((err ** 2).sum() / err.shape[0])

def compute_errors(model, eids, X, y, nb):
    return y - ipr_predict(model, eids, X, nb)

def compute_rmse(model, eids, X, y, nb):
    errors = compute_errors(model, eids, X, y, nb)
    return rmse_from_err(errors)


def make_parser():
    parser = argparse.ArgumentParser(
        description="mixed-membership multi-linear regression")
    parser.add_argument(
        '-m', '--method',
        choices=('sgd', 'als'), default='sgd',
        help='learning method to use')
    parser.add_argument(
        '-tr', '--train',
        help='path of training data file in csv format')
    parser.add_argument(
        '-te', '--test',
        help='path of test data file in csv format')
    parser.add_argument(
        '-l', '--nmodels',
        type=int, default=3,
        help='number of linear regression models')
    parser.add_argument(
        '-lw', '--lambda-w',
        type=float, default=0.01,
        help='regularization multiplier for P and W')
    parser.add_argument(
        '-lb', '--lambda-b',
        type=float, default=0.001,
        help='regularization multiplier for s and c')
    parser.add_argument(
        '--bounds', default='0,4',
        help='upper,lower bound for rating bounding')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-i', '--iters',
        type=int, default=10,
        help='number of iterations')
    parser.add_argument(
        '-s', '--init-std',
        type=float, default=(1. / 2 ** 4),
        help='standard deviation of Gaussian noise used in model param'
             'initialization')
    parser.add_argument(
        '-t', '--target',
        default='grade',
        help='target variable to predict; default is "grade"')
    parser.add_argument(
        '-n', '--nonneg',
        action='store_true', default=False,
        help='enable non-negativity constraints on all params')
    parser.add_argument(
        '-v', '--verbose',
        type=int, default=0, choices=(0, 1, 2),
        help='verbosity level; 0=None, 1=INFO, 2=DEBUG')
    # parser.add_argument(
    #     '-o', '--output',
    #     default='',
    #     help='directory to save model params to; default is none: do not save')
    parser.add_argument(
        '-f', '--feature-guide',
        help='file to specify target, categorical, and real-valued features; '
             'see the docstring for more detailed info on the format')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose == 2 else
               logging.INFO if args.verbose == 1 else
               logging.ERROR),
        format="[%(asctime)s]: %(message)s")

    # Sanity check bounds argument.
    bounds = args.bounds.split(',')
    if len(bounds) != 2:
        print 'bounds must be comma-separated, got %s' % args.bounds
        sys.exit(BOUNDS_FORMAT)

    logging.info('reading train/test files')
    try:
        eids, X, y, test_eids, test_X, test_y, f_indices, nb = \
            read_data(args.train, args.test, args.feature_guide)
        errno = None
    except IOError as e:
        errno = BAD_FILENAME
    except BadFeatureConfig as e:
        errno = BAD_FEATURE_CONF

    if errno is not None:
        logging.error(e.message)
        sys.exit(errno)

    # Extract model parameters from cmdline args.
    k=args.nmodels
    lambda_w=args.lambda_w
    lambda_b=args.lambda_b
    iters=args.iters
    std=args.init_std
    nn=args.nonneg
    verbose=args.verbose

    b1 = np.unique(eids).shape[0]  # num unique values for entity to profile
    n, nf = X.shape  # num training examples and num features
    p = nf - nb  # num non-entity predictor variables

    # Init params.
    w0 = 0

    dtype = np.float64
    # P_zeros = np.zeros((b1, k)).astype(dtype)
    # W_zeros = np.zeros((k, p)).astype(dtype)

    w = np.zeros(nb).astype(dtype)
    P = np.random.normal(0, std, (b1, k)).astype(dtype)
    W = np.random.normal(0, std, (k, p)).astype(dtype)

    model = {
        'w0': w0,
        'w': w,
        'P': P,
        'W': W
    }

    # Data setup.
    y = y.astype(dtype)
    X = X.tocsc().astype(dtype)
    B_t = X[:, :nb]  # n x nb
    X_t = X[:, nb:]  # n x p
    X_tsq = X_t.copy()
    X_tsq.data = X_tsq.data ** 2

    # Precompute error.
    err = compute_errors(model, eids, X, y, nb)
    logging.info('initial RMSE:\t%.4f' % rmse_from_err(err))

    # Update w0. Same as w0 = y.mean() on first iteration.
    w0_new = (err - w0).sum() / n
    err += (w0 - w0_new)
    w0 = w0_new

    model['w0'] = w0
    logging.info('Train RMSE after w0:\t%.4f' % compute_rmse(
        model, eids, X, y, nb))

    # Set lrate; necessary to avoid overzealous steps. Why?
    lrate = 0.01

    start = time.time()
    logging.info('training model for %d iterations' % iters)
    for inum in range(iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (inum + 1, elapsed))

        # Update w; all bias terms are 0 or 1, so squaring does nothing.
        B_sum = np.asarray(B_t.sum(axis=0))[0]
        err_sum = err * B_t
        reg_sum = w * B_sum
        w_new = ((err_sum - reg_sum) / B_sum) * lrate
        err += B_t * (w - w_new)

        # Recompute error to avoid rounding errors.
        logging.info('Train RMSE cache:  \t%.4f' % rmse_from_err(err))
        err = compute_errors(model, eids, X, y, nb)
        logging.info('Train RMSE after w:\t%.4f' % rmse_from_err(err))

        # Update all regression coefficients W.
        membs = P[eids]
        sq_sum = (membs ** 2).T * X_tsq
        err_sum = (membs * err[:, np.newaxis]).T * X_t
        W_new = lrate * (W * sq_sum - err_sum) / (lambda_w - sq_sum)
        err += (X_t * W.T * membs).dot(W - W_new).sum(axis=1)

        # Recompute error to avoid rounding errors.
        logging.info('Train RMSE cache:  \t%.4f' % rmse_from_err(err))
        err = compute_errors(model, eids, X, y, nb)
        logging.info('train RMSE after W:\t%.4f' % rmse_from_err(err))

        # Update memberships for all i primary entity values.
        tmp = X_t.dot(W.T)
        reg_coeffs = X_tsq.dot((W ** 2).T)
        sq_sum = reg_coeffs.sum(axis=0)
        for i in xrange(b1):
            P_i = P[i]
            err_sum = W.dot(err * X_t)
            Pi_new = lrate * (reg_coeffs.dot(P_i).sum() - err_sum) / (lambda_w - sq_sum)
            err += tmp.dot(P_i - Pi_new)

        # recompute err to correct rounding errors.
        logging.info('Train RMSE cache:  \t%.4f' % rmse_from_err(err))
        err = compute_errors(model, eids, X, y, nb)
        logging.info('Train RMSE after P:\t%.4f' % rmse_from_err(err))


    elapsed = time.time() - start
    logging.info('total time elapsed:\t(%.2fs)' % elapsed)

    logging.info('making predictions')
    err = ipr_predict(model, test_eids, test_X.tocsc(), nb) - test_y
    print 'FM RMSE:\t%.4f' % rmse_from_err(err)

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)
