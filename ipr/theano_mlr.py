"""
Mixed-membership Multi-Linear Regression model implemented with Theano.

"""
import os
import sys
import time
import logging
import argparse

from itertools import izip

import theano
import theano.tensor as T
import numpy as np
import pandas as pd
from sklearn import preprocessing

from util import save_np_vars, add_squared_features


def rmse_from_err(err):
    return np.sqrt((err ** 2).sum() / len(err))

def fro_norm(mat):
    return (mat ** 2).sum()

def l2_norm(mat):
    return T.sqrt((mat ** 2).sum())

def l1_norm(mat):
    return abs(mat).sum()


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


def shared_dataset(dataset):
    """Convert dataset to Theano shared vars to reduce GPU overhead."""
    shvar = lambda var: theano.shared(
        np.asarray(var, dtype=theano.config.floatX))
    return map(shvar, dataset)


class MLR(object):
    """Mixed-membership multi-linear regression model."""

    def __init__(self, n, m, nf, l, lrate, iters, std=0.01):
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

        # Initialize parameters using Gaussian noise for initial values.
        self.reinit_params(std)

        # Symbolic variables for prediction/evaluation functions.
        self._X = T.fmatrix('X')
        self._uids = T.ivector('uids')
        self._iids = T.ivector('iids')
        self._y = T.fvector('y')

        loss = lambda x_i, _u, _i: (
            self.b_s[_u] + self.b_c[_i] + self.P[_u].T.dot(self.W).dot(x_i))

        logging.info('compiling prediction function')
        self._predicted, updates = theano.scan(
            fn=loss, sequences=[self._X, self._uids, self._iids])
        self.make_predictions = theano.function(
            inputs=[self._X, self._uids, self._iids],
            outputs=self._predicted, allow_input_downcast=True)

        logging.info('compiling error computation function')
        self._prediction_err = self._predicted.reshape(self._y.shape) - self._y
        self.compute_err = theano.function(
            inputs=[self._X, self._y, self._uids, self._iids],
            outputs=self._prediction_err, allow_input_downcast=True)

        logging.info('compiling rmse computation function')
        self._rmse = T.sqrt((self._prediction_err ** 2).sum() /
                             self._prediction_err.shape[0])
        self.compute_rmse = theano.function(
            inputs=[self._X, self._y, self._uids, self._iids],
            outputs=self._rmse, allow_input_downcast=True)

    def reinit_params(self, std):
        """Set up Theano variables for incremental SGD learning.

        b_s  = (n)         # student bias term
        b_c  = (m)         # course bias term
        P    = (n x l x 1) # student membership vector
        W    = (l x nf)    # regression coefficient matrix

        """
        randn = lambda dim: np.random.normal(0.01, std, dim)
        self.b_s = theano.shared(
            value=randn(self.n), name='b_s', borrow=True)
        self.b_c = theano.shared(
            value=randn(self.m), name='b_c', borrow=True)
        self.P = theano.shared(
            value=randn((self.n, self.l, 1)), name='P', borrow=True)
        self.W = theano.shared(
            value=randn((self.l, self.nf)), name='W', borrow=True)

        self._P_zeros = theano.shared(value=np.zeros((self.l, 1)))
        self._W_zeros = theano.shared(value=np.zeros((self.l, self.nf)))

    @property
    def params(self):
        return [self.b_s, self.b_c, self.P, self.W]

    @property
    def param_values(self):
        return {
            's': self.b_s.get_value(),
            'c': self.b_c.get_value(),
            'P': self.P.get_value(),
            'W': self.W.get_value()
        }

    def log_shared(self):
        for var in self.params:
            logging.debug('%s\n%s' % (var.name, str(var.get_value())))

    def fit(self, X, y, uids, iids, norm='fro'):
        """Fit the model to the data."""

        # Optimize CPU -> GPU memory transfer.
        sX, sy, suids, siids = shared_dataset((X, y, uids, iids))

        # Indices for users/items
        i = T.lscalar('index')
        uid_i = T.cast(suids[i], 'int32')
        iid_i = T.cast(siids[i], 'int32')

        # Define error function.
        g_hat = (self.b_s[uid_i] + self.b_c[iid_i] +
                 self.P[uid_i].T.dot(self.W).dot(sX[i]))
        error = (g_hat - sy[i]) ** 2  # for squared loss

        # Define regularization function.
        norm = (fro_norm if norm == 'fro' else
                l2_norm if norm == 'l2' else
                l1_norm)
        reg = args.lambda_ * (norm(self.P[uid_i]) + norm(self.W))

        # The loss function minimizes least squares with regularization.
        loss = (error + reg).sum()  # the sum just pulls out the single value

        logging.info('compiling compute code')
        gradients = dict(zip(
            [v.name for v in self.params], T.grad(loss, self.params)))
        inputs = [i]

        # Define model training functions.
        P_update = T.set_subtensor(
            self.P[uid_i],
            T.maximum(
                self._P_zeros,
                self.P[uid_i] - self.lrate * gradients[self.P.name][uid_i]))
        train_P = theano.function(
            inputs=inputs, outputs=loss, name='train_P',
            updates=[(self.P, P_update)])

        W_update = T.maximum(
            self._W_zeros, self.W - self.lrate * gradients[self.W.name])
        train_W = theano.function(
            inputs=inputs, outputs=loss, name='train_W',
            updates=[(self.W, W_update)])

        bias_update = lambda bvar, idvar: (
            T.set_subtensor(
                bvar[idvar],
                T.largest(
                    0,
                    bvar[idvar] - self.lrate * gradients[bvar.name][idvar])))
        b_s_update = bias_update(self.b_s, uid_i)
        b_c_update = bias_update(self.b_c, iid_i)
        train_B = theano.function(
            inputs=inputs, outputs=loss, name='train_B',
            updates=[(self.b_s, b_s_update), (self.b_c, b_c_update)])

        # Main training loop.
        indices = range(len(X))
        logging.info('training model for %d iterations' % self.iters)
        start = time.time()
        for it in range(args.iters):
            elapsed = time.time() - start
            logging.info('iteration %03d\t(%.2fs)' % (it + 1, elapsed))
            for idx in np.random.permutation(indices):
                train_P(idx)
                train_B(idx)
                train_W(idx)

            if args.verbose >= 1:
                self.log_rmse('Train', X, y, uids, iids)

            self.log_shared()

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
        default='',
        help='directory to save model params to; default is none: do not save')
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
    # features = ['term', 'gender', 'age', 'schrs', 'hsgpa', 'cum_gpa',
    #             'cum_cgpa', 'chrs', 'term_chrs', 'term_enrolled']
    data_keys = [uid, iid, args.target]
    # to_read = list(set(features + data_keys))

    logging.info('reading train/test data')
    # logging.info('reading columns: %s' % ', '.join(to_read))
    # data = pd.read_csv(args.data_file, usecols=to_read)
    data = pd.read_csv(args.data_file)
    features = list(set(data.columns) - set(data_keys))
    logging.info('read columns: %s' % ', '.join(data.columns))

    # Add quadratic features of power 2.
    if args.quadratic:
        features += add_squared_features(data, features)

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

    # Z-score scaling to mean of 0 and variance of 1.
    # scale_features(data, features)
    scaler = preprocessing.StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    # Get dimensions of training data.
    nd, nf = train_x.shape
    l = args.nmodels

    logging.info('%d users, %d items' % (n, m))
    logging.info('%d dyads with %d features' % (nd, nf))
    logging.info('l=%d, lr=%f' % (l, args.lrate))

    model = MLR(n, m, nf, l, args.lrate, args.iters)
    model.fit(train_x, train_y, train_uids, train_iids)

    logging.info('making predictions')
    print 'MLR RMSE:\t%.4f' % model.compute_rmse(
        test_x, test_y, test_uids, test_iids)

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(train_y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)

    # Save model params.
    if args.output:
        save_np_vars(model.param_values, args.output)
