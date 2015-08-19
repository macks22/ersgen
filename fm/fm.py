"""
Implement Factorization Machine (FM) model.

The following learning algorithms have been implemented:

1.  Alternating Least Squares (ALS)

The command line expects data in CSV format, with separate files for training
and test records. A required file specifies the format of the attributes in the
CSV files. Three types of attributes are delineated: (1) target, (2)
categorical, and (3) real-valued. The target attribute is separated from the
others during training and predicted for the test data after the model is
learned. The categorical attributes are one-hot encoded, and the real-valued
attributes are scaled using Z-score scaling (0 mean, unit variance).

The format for the features file is as follows:

    t:<target>;
    c:<comma-separated categorical variable names>;
    r:<comma-separated real-valued variable names>;

Whitespace is ignored, as are lines that start with a "#" symbol. Any variables
not included in one of the three groups are ignored. They are used neither for
training nor prediction.

"""
import sys
import logging
import argparse
import itertools as it

import numpy as np
import scipy as sp
import pandas as pd
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
        c:<comma-separated categorical variable names>;
        r:<comma-separated real-valued variable names>;

    Whitespace is ignored, as are lines that start with a "#" symbol. Any
    variables not included in one of the three groups is ignored.

    Args:
        fname (str): Path of the file containing the feature guide.

    Returns:
        iterable: (target, categoricals list, real-valued list).

    """
    with open(fname) as f:
        lines = [l.strip() for l in f.read().split('\n')
                 if not l.startswith('#') and l.strip()]

    parsing = False
    keys = ['t', 'c', 'r']
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

    num_features = len(vars['c']) + len(vars['r'])
    if not num_features > 0:
        raise BadFeatureConfig('no predictors specified')

    spec = (vars['t'][0], vars['c'], vars['r'])
    target, cats, reals = spec
    logging.info('read the following feature guide:')
    logging.info('target: %s' % target)
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
        tuple: (train_X, train_y, test_X, test_y), preprocessed train/test data.

    """
    try:
        target, cats, reals = read_feature_guide(conf_file)
    except IOError:
        raise IOError('invalid feature guide conf file path: %s' % conf_file)

    to_read = [target] + cats + reals
    def read_file(name, fpath):
        try:
            data = pd.read_csv(fpath, usecols=to_read)
        except IOError:
            raise IOError('invalid %s file path: %s' % (name, fpath))
        except ValueError as err:
            attr_name = err.args[0].split("'")[1]
            attr_type = ('categorical' if attr_name in cats else
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

    # Z-score scaling of real-valued features.
    scaler = preprocessing.StandardScaler()
    train_reals = scaler.fit_transform(train[reals])
    test_reals = scaler.transform(test[reals])

    # One-hot encoding of categorical features.
    all_cats = pd.concat((train[cats], test[cats]))
    encoder = preprocessing.OneHotEncoder()
    enc_cats = encoder.fit_transform(all_cats)
    train_cats = enc_cats[:nd_train]
    test_cats = enc_cats[nd_train:]

    logging.info('after one-hot encoding, found # unique values:')
    for attr, n_values in zip(cats, encoder.n_values_):
        logging.info('%s: %d' % (attr, n_values))

    # Put all features together.
    train_X = sp.sparse.hstack((train_cats, train_reals))
    test_X = sp.sparse.hstack((test_cats, test_reals))
    logging.info('Total of %d features after encoding' % train_X.shape[1])

    return train_X, train_y, test_X, test_y


def predict(X, w0, w, V):
    """Predict y values for data X given model params w0, w, and V.

    Args:
        X (sp.sparse.coo.coo_matrix): Sparse data matrix with instances as rows.
        w0 (float): Global bias term.
        w (np.ndarray[np.double_t, ndim=1]): 1-way interaction terms.
        V (np.ndarray[np.double_t, ndim=2]): 2-way interaction terms.

    Returns:
        np.ndarray[np.double_t, ndim=1]: Predictions \hat{y}.

    """
    N = X.shape[0]
    predictions = np.zeros(N) + w0

    two_way = np.zeros(N)
    for f in xrange(V.shape[1]):
        t1 = np.zeros(N)
        t2 = np.zeros(N)
        v_f = V[:, f]
        for i, j, x in it.izip(X.row, X.col, X.data):
            tmp = v_f[j] * x
            t1[i] += tmp
            t2[i] += tmp ** 2

        two_way += t1 ** 2 - t2
    two_way *= 0.5

    one_way = np.zeros(N)
    for i, j, x in it.izip(X.row, X.col, X.data):
        one_way[i] += w[j] * x

    predictions += one_way + two_way
    return predictions


def make_parser():
    parser = argparse.ArgumentParser(
        description='BPMLR for non-cold-start dyadic response prediction')
    parser.add_argument(
        '-v', '--verbose',
        type=int, choices=(0, 1, 2), default=1,
        help='enable verbose logging output')
    parser.add_argument(
        '-tr', '--train',
        help='path of training data file in csv format')
    parser.add_argument(
        '-te', '--test',
        help='path of test data file in csv format')
    parser.add_argument(
        '-k', '--dim',
        type=int, default=8,
        help='dimensionality of factorized interactions')
    parser.add_argument(
        '-i', '--iterations',
        type=int, default=50,
        help='number of iterations to run for')
    parser.add_argument(
        '--bounds', default='0,4',
        help='upper,lower bound for rating bounding')
    parser.add_argument(
        '-std', '--init-stdev',
        type=float, default=0.01,
        help='standard deviation of Gaussian noise initialization for'
             'factorized terms')
    parser.add_argument(
        '-s', '--stopping-threshold',
        type=float, default=0.000001,
        help='early stopping threshold')
    parser.add_argument(
        '-l', '--lambda_',
        default='0.01,0.1',
        help='1-way and 2-way regularization terms, comma-separated; '
             'defaults to 0.01,0.1')
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

    # Sanity check regularization terms argument.
    try:
        lambda_w, lambda_v = map(float, args.lambda_.split(','))
    except ValueError:
        print 'invalid regularization param: %s' % args.lambda_
        sys.exit(BAD_CMDLINE_ARG)

    logging.info('reading train/test files')
    try:
        X, y, test_X, test_y = read_data(
            args.train, args.test, args.feature_guide)
        errno = None
    except IOError as e:
        errno = BAD_FILENAME
    except BadFeatureConfig as e:
        errno = BAD_FEATURE_CONF

    if errno is not None:
        logging.error(e.message)
        sys.exit(errno)

    # We have the data, let's begin.
    nd, nf = X.shape
    k = args.dim
    X_csc = X.tocsc()  # for sparse column indexing
    X_T = X_csc.T

    # Init w0, w, and V.
    w0 = 0
    w = np.zeros(nf)
    V = np.random.normal(0, args.init_stdev, (nf, k))

    # Precompute e and q.
    y_hat = predict(X, w0, w, V)
    e = y_hat - y

    q = np.zeros((nd, k))
    for f in xrange(k):
        for i, j, x_j in it.izip(X.row, X.col, X.data):
            q[i, f] += V[j, f] * x_j

    trace = {
        'w0': np.zeros(args.iterations),
        'w':  np.zeros((args.iterations, nf)),
        'V':  np.zeros((args.iterations, nf, k))
    }

    # Set learning rate for 2-way interactions.
    lrate = 0.02

    # Main optimization loop.
    prev_rmse = np.sqrt((e ** 2).sum() / nd)
    logging.info('initial RMSE: %.4f' % prev_rmse)
    for iteration in xrange(args.iterations):

        # Learn global bias term.
        w0_new = (e - w0).sum() / nd
        e += w0_new - w0
        w0 = w0_new

        trace['w0'][iteration] = w0

        # Learn 1-way interaction terms.
        for j, col in enumerate(X_T):
            w1 = ((e[col.indices] - w[j] * col.data) * col.data).sum()
            w2 = (col.data ** 2).sum()
            w_new = -(w1 / (w2 + lambda_w))
            e[col.indices] += (w_new - w[j]) * col.data
            w[j] = w_new

        trace['w'][iteration] = w

        # TODO: the error only starts increasing during this loop.
        # e gets screwed up somehow, so q probably does too.
        # In particular, e goes negative, which is not good. The v_new updates
        # are too aggressive, or the error update rule is wrong.
        # It could also be simply that the rounding error is large.
        # Learn 2-way interaction terms.
        for f in xrange(k):
            q_f = q[:, f]
            v_f = V[:, f]
            for j, col in enumerate(X_T):
                rows = col.indices
                v_jf = v_f[j]

                h = col.data * (q_f[rows] - col.data * v_jf)
                # sum_nominator = ((e[rows] - v_jf * h) * h).sum()
                sum_nominator = ((v_jf * h - e[rows]) * h).sum()
                sum_denominator = (h ** 2).sum()

                v_new = (sum_nominator / (sum_denominator + lambda_v)) * lrate
                update = (v_new - v_jf) * col.data
                e[rows]    += update
                q[rows, f] += update
                V[j, f] = v_new

        trace['V'][iteration] = V

        # Re-evaluate RMSE to prepare for stopping check.
        # Also recompute e to avoid gradual numerical rounding errors.
        y_hat = predict(X, w0, w, V)
        e = y_hat - y
        rmse = np.sqrt((e ** 2).sum() / nd)

        logging.info('RMSE after iteration %02d: %.4f' % (iteration, rmse))
        if prev_rmse - rmse < args.stopping_threshold:
            logging.info('stopping threshold reached')
            break
        else:
            prev_rmse = rmse
