import sys
import logging
import argparse

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import preprocessing

from cfm import fit_fm_als, predict, rmse_from_err


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

    # Create a feature map for decoding one-hot encoding.
    nreal = train_reals.shape[1]
    ncats = encoder.active_features_.shape[0]
    nf = nreal + ncats

    counts = np.array([
        all_cats[cats[i]].unique().shape[0]
        for i in xrange(len(cats))
    ])
    indices = zip(cats, np.cumsum(counts))
    indices += zip(reals, range(indices[-1][1] + 1, nf + 1))

    logging.info('after one-hot encoding, found # unique values:')
    for attr, n_values in zip(cats, counts):
        logging.info('%s: %d' % (attr, n_values))

    logging.info('number of active categorical features: %d of %d' % (
        ncats, encoder.n_values_.sum()))
    logging.info('number of real-valued features: %d' % nreal)

    # Put all features together.
    train_X = sp.sparse.hstack((train_cats, train_reals))
    test_X = sp.sparse.hstack((test_cats, test_reals))
    logging.info('Total of %d features after encoding' % nf)

    return train_X, train_y, test_X, test_y, indices


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
        type=float, default=0.001,
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
        X, y, test_X, test_y, f_indices = read_data(
            args.train, args.test, args.feature_guide)
        errno = None
    except IOError as e:
        errno = BAD_FILENAME
    except BadFeatureConfig as e:
        errno = BAD_FEATURE_CONF

    if errno is not None:
        logging.error(e.message)
        sys.exit(errno)

    w0, w, V = fit_fm_als(X, y,
                          iters=args.iterations,
                          threshold=args.stopping_threshold,
                          k=args.dim,
                          lambda_w=lambda_w,
                          lambda_v=lambda_v,
                          init_stdev=args.init_stdev)


    logging.info('making predictions')
    e = predict(test_X, w0, w, V) - test_y
    rmse = rmse_from_err(e)
    print 'FM RMSE:\t%.4f' % rmse

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)

    # Calculate feature importance metrics.
    nd, nf = X.shape
    X = abs(X.tocsc())
    w = abs(w).reshape(w.shape[0], 1)
    sigma1 = np.zeros((nd, nf))
    sigma2 = np.zeros((nd, nf))
    Z = abs(V.dot(V.T))

    # Compute sigma1 and sigma2 for all j.
    X_T = X.T
    z_diag = np.diag(Z) / 2
    for j in xrange(nf):
        col = X_T[j]
        dat = col.data[:, np.newaxis]
        rows = col.indices

        sigma1[rows, j] = np.asarray(dat * w[j]).squeeze()

        col_sq = dat ** 2
        top = dat * Z[j]
        bot = dat + X[rows]
        sigma2[rows, j] = np.asarray(
            np.multiply(col_sq, ((top / bot).sum(axis=1) - z_diag[j])))\
                .squeeze()

    # Compute T_d terms and f.
    T = sigma1.sum(axis=1) + sigma2.sum(axis=1)
    f = (sigma1.sum(axis=0) + sigma2.sum(axis=0)) / T.sum()
    assert(np.isclose(f.sum(), 1.0))

    # Finally, combine one-hot encoded importances to get final importances.
    I = {}
    prev = 0
    for i in xrange(len(f_indices)):
        fname, last = f_indices[i]
        I[fname] = f[prev: last].sum()
        prev = last

    # Plot feature importance.
    colname = 'Importance'
    I = pd.DataFrame(I.values(), index=I.keys(), columns=[colname])\
          .sort(colname, ascending=False)

    deep_blue = sns.color_palette('colorblind')[0]
    ax = sns.barplot(data=I, x=colname, y=I.index, color=deep_blue)
    ax.set(title='Feature Importance for Grade Prediction',
           ylabel='Feature',
           xlabel='Proportion of Deviation From Intercept')
    ax.figure.show()
