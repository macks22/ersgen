"""
Mixed-membership Multi-Linear Regression model (non-Theano version).

"""
import logging
import argparse

import numpy as np
import pandas as pd

from cmlr import compute_rmse, fit_mlr, rmse_from_err


def scale_features(data, features):
    """Scale all `features` using Z-score scaling."""
    logging.info('performing z-score scaling on features')
    logging.debug('%s' % ', '.join(features))
    for f in features:
        data[f] = (data[f] - data[f].mean()) / data[f].std(ddof=0)


def add_squared_features(data, features):
    """Add squared versions of the given features."""
    logging.info('adding %d quadratic features' % len(features))
    new_keys = []
    for f in features:
        key = '%s^2' % f
        data[key] = data[f] ** 2
        new_keys.append(key)
    return new_keys


def map_ids(data, key):
    """Map ids to 0-contiguous index. This enables the use of these ids as
    indices into an array (for the bias terms, for instance). This returns the
    number of unique IDs for `key`.
    """
    ids = data[key].unique()
    n = len(ids)
    id_map = dict(zip(ids, range(n)))
    data[key] = data[key].apply(lambda _id: id_map[_id])
    return n


def split_train_test(data, term):
    return data[data.term < term], data[data.term == term]


def make_parser():
    parser = argparse.ArgumentParser(
        description="mixed-membership multi-linear regression")
    parser.add_argument(
        '-d', '--data_file', default='',
        help='path of data file')
    parser.add_argument(
        '-l', '--nmodels',
        type=int, default=3,
        help='number of linear regression models')
    parser.add_argument(
        '-lam', '--lambda_',
        type=float, default=0.01,
        help='regularization multiplier')
    parser.add_argument(
        '-r', '--regularization',
        choices=('fro', 'l1', 'l2'), default='fro',
        help='type of regularization to use')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-i', '--iters',
        type=int, default=10,
        help='number of iterations')
    parser.add_argument(
        '-s', '--std',
        type=float, default=0.01,
        help='standard deviation of Gaussian noise used in model param'
             'initialization')
    parser.add_argument(
        '-q', '--quadratic',
        action='store_true', default=False,
        help='add quadratic features (power 2)')
    parser.add_argument(
        '-t', '--target',
        default='grade',
        help='target variable to predict; default is "grade"')
    parser.add_argument(
        '-v', '--verbose',
        type=int, default=0, choices=(0, 1, 2),
        help='verbosity level; 0=None, 1=INFO, 2=DEBUG')
    parser.add_argument(
        '-o', '--output',
        action='store_true', default=False)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose == 2 else
               logging.INFO if args.verbose == 1 else
               logging.ERROR),
        format="[%(asctime)s]: %(message)s")

    uid = 'sid'
    iid = 'cid'
    features = ['term', 'gender', 'age', 'schrs', 'hsgpa', 'cum_gpa',
                'cum_cgpa', 'chrs', 'term_chrs', 'term_enrolled']
    data_keys = [uid, iid, args.target]
    to_read = list(set(features + data_keys))

    logging.info('reading train/test data')
    logging.info('reading columns: %s' % ', '.join(to_read))
    data = pd.read_csv(args.data_file, usecols=to_read)

    # Add quadratic features of power 2.
    if args.quadratic:
        features += add_squared_features(data, features)

    # Z-score scaling to mean of 0 and variance of 1.
    scale_features(data, features)

    # Map user/item ids to bias indices.
    n = map_ids(data, uid)
    m = map_ids(data, iid)

    # Split dataset into train & test, predicting only for last term (for now).
    train, test = split_train_test(data, data.term.max())

    def split_data(df):
        """Split data into X, y, uids, iids."""
        return (df.drop(data_keys, axis=1).values,
                df[args.target].values,
                df[uid].values,
                df[iid].values)

    logging.info('splitting train/test sets into X, y')
    train_x, train_y, train_uids, train_iids = split_data(train)
    test_x, test_y, test_uids, test_iids = split_data(test)

    # Get dimensions of training data.
    nd, nf = train_x.shape
    l = args.nmodels

    logging.info('%d users, %d items' % (n, m))
    logging.info('%d dyads with %d features' % (nd, nf))
    logging.info('l=%d, lr=%f' % (l, args.lrate))

    # Fit model.
    model = fit_mlr(train_x, train_y, train_uids, train_iids,
                    l=l,
                    lrate=args.lrate,
                    lambda_=args.lambda_,
                    iters=args.iters,
                    std=args.std,
                    verbose=args.verbose)

    logging.info('making predictions')
    print 'MLR RMSE:\t%.4f' % compute_rmse(
        model, test_x, test_y, test_uids, test_iids)

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(train_y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)
