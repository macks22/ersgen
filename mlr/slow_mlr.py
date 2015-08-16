"""
Mixed-membership Multi-Linear Regression model (non-Theano version).

"""
import os
import sys
import time
import logging
import argparse

from itertools import izip

import numpy as np
import pandas as pd


def rmse_from_err(err):
    return np.sqrt((err ** 2).sum() / len(err))

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


class MLR(object):
    """Mixed-membership multi-linear regression model."""

    def __init__(self, n, m, nf, l, lrate, iters, _lambda, std=0.01):
        """Initialize model parameters and compile prediction and eval
        functions.

        n =   # students (subscript s)
        m =   # courses  (subscript c)
        l =   # models   (subscript d)
        nf =  # features (subscript k)
        nd =  # dyads    (subscript i) = n x m

        """
        self.n = n
        self.m = m
        self.l = l
        self.nf = nf
        self.lrate = lrate
        self.iters = iters
        self._lambda = _lambda

        # Initialize parameters using Gaussian noise for initial values.
        self.reinit_params(std)

    def reinit_params(self, std):
        """Set up variables for incremental SGD learning.

        b_s  = (n)         # student bias term
        b_c  = (m)         # course bias term
        P    = (n x l x 1) # student membership vector
        W    = (l x nf)    # regression coefficient matrix

        """
        randn = lambda dim: np.random.normal(0.01, std, dim)
        self.b_s = randn(self.n)
        self.b_c = randn(self.m)
        self.P = randn((self.n, self.l, 1))
        self.W = randn((self.l, self.nf))

    @property
    def params(self):
        return (
            ('b_s', self.b_s),
            ('b_c', self.b_c),
            ('P',   self.P),
            ('W',   self.W)
        )

    def log_params(self):
        for name, var in self.params:
            logging.debug('%s\n%s' % (name, str(var)))

    def compute_errors(self, X, y, uids, iids):
        if hasattr(X, 'values'):
            X = X.values

        if hasattr(y, 'values'):
            y = y.values

        n = len(X)
        b_s = self.b_s[uids]
        b_c = self.b_c[iids]
        P = self.P[uids]

        return np.array([
            b_s[i] + b_c[i] + P[i].T.dot(self.W).dot(X[i]) - y[i]
            for i in xrange(n)
        ])

    def compute_rmse(self, X, y, uids, iids):
        errors = self.compute_errors(X, y, uids, iids)
        return rmse_from_err(errors)

    def fit(self, X, y, uids, iids, norm='fro'):
        """Fit the model to the data."""
        logging.info('training model for %d iterations' % self.iters)
        if hasattr(X, 'values'):
            X = X.values

        X = X.reshape(list(X.shape) + [1])

        indices = range(len(X))
        start = time.time()
        for _iter in range(self.iters):
            elapsed = time.time() - start
            logging.info('iteration %03d\t(%.2fs)' % (_iter + 1, elapsed))
            for _sc in np.random.permutation(indices):
                s = uids[_sc]
                c = iids[_sc]
                P_s = self.P[s]

                # compute error
                y_hat = (self.b_s[s] + self.b_c[c] +
                         P_s.T.dot(self.W).dot(X[_sc]))
                error = self.lrate * 2 * (y_hat - y[_sc])

                # update parameters
                self.P[s] -= (error * self.W.dot(X[_sc]) +
                              2 * self._lambda * P_s)
                self.b_s[s] -= error
                self.b_c[c] -= error
                self.W -= (error * P_s.dot(X[_sc].T) +
                           2 * self._lambda * self.W)

            if args.verbose >= 1:
                self.log_rmse('Train', X, y, uids, iids)

            self.log_params()

        elapsed = time.time() - start
        logging.info('total time elapsed:\t(%.2fs)' % elapsed)

    def log_rmse(self, name, X, y, uids, iids):
        logging.info('%s RMSE:\t%.4f' % (
            name, self.compute_rmse(X, y, uids, iids)))


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
    data_file = (args.data_file if args.data_file else
                 'data-n500-m50-t4-d5538.csv')
    data = pd.read_csv(data_file, usecols=to_read)

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
        return (df.drop(data_keys, axis=1), df[args.target], df[uid], df[iid])

    logging.info('splitting train/test sets into X, y')
    train_x, train_y, train_uids, train_iids = split_data(train)
    test_x, test_y, test_uids, test_iids = split_data(test)

    # Get dimensions of training data.
    nd, nf = train_x.shape
    l = args.nmodels

    logging.info('%d users, %d items' % (n, m))
    logging.info('%d dyads with %d features' % (nd, nf))
    logging.info('l=%d, lr=%f' % (l, args.lrate))

    model = MLR(n, m, nf, l, args.lrate, args.iters, args.lambda_, args.std)
    model.fit(train_x, train_y, train_uids, train_iids)

    logging.info('making predictions')
    print 'MLR RMSE:\t%.4f' % model.compute_rmse(
        test_x, test_y, test_uids, test_iids)

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(train_y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)
