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


def rmse_from_err(err):
    return np.sqrt((err ** 2).sum() / len(err))

def fro_norm(mat):
    return (mat ** 2).sum()

def l2_norm(mat):
    return T.sqrt((mat ** 2).sum())

def l1_norm(mat):
    return abs(mat).sum()


def scale_features(data, features):
    """Scale all `features` using Z-score scaling."""
    logging.info('performing z-score scaling on features')
    logging.debug('%s' % ', '.join(features))
    for f in features:
        data[f] = (data[f] - data[f].mean()) / data[f].std(ddof=0)


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
        '-n', '--nmodels',
        type=int, default=3,
        help='number of linear regression models')
    parser.add_argument(
        '-l', '--lambda_',
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

    # Z-score scaling to mean of 0 and variance of 1.
    scale_features(data, features)

    # Add quadratic features of power 2.
    if args.quadratic:
        logging.info('adding %d quadratic features' % len(features))
        for f in features:
            key = '%s^2' % f
            data[key] = data[f] ** 2
            data[key] = (data[key] - data[key].mean()) / data[key].std(ddof=0)

    # Map user/item ids to bias indices.
    n = map_ids(data, uid)
    m = map_ids(data, iid)

    # Split dataset into train & test, predicting only for last term (for now).
    train, test = split_train_test(data, data.term.max())

    def split_data(df):
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

    # Set up Theano variables for incremental SGD learning.
    """
    nf =  # features (subscript k)
    n =   # students (subscript s)
    m =   # courses  (subscript c)
    l =   # models   (subscript d)
    nd =  # dyads    (subscript i) = n x m

    b_s  = (1)       # student bias term
    b_c  = (1)       # course bias term
    p_s  = (l x 1)   # student membership vector
    W    = (l x nf)  # regression coefficient matrix
    f_sc = (nf x 1)  # dyad feature vector

    g = (1)                                     # actual grade
    g_hat = b_s + b_c + p_s.T.dot(W).dot(f_sc)  # predicted grade
    """
    randn = lambda dim: np.random.normal(0.01, 0.01, dim)
    b_s = theano.shared(value=randn(n), name='b_s', borrow=True)
    b_c = theano.shared(value=randn(m), name='b_c', borrow=True)
    P = theano.shared(value=randn((n, l, 1)), name='P', borrow=True)
    P_zeros = theano.shared(value=np.zeros((l, 1)))
    W = theano.shared(value=randn((l, nf)), name='W', borrow=True)
    W_zeros = theano.shared(value=np.zeros((l, nf)))
    f_sc = T.dvector('f_sc')  # (nf, 1)

    def log_shared():
        logging.debug('%s:\n%s' % (b_s.name, str(b_s.get_value())))
        logging.debug('%s:\n%s' % (b_c.name, str(b_c.get_value())))
        logging.debug('%s:\n%s' % (P.name, str(P.get_value())))
        logging.debug('%s:\n%s' % (W.name, str(W.get_value())))

    log_shared()

    # Indices for users/items
    _s = T.lscalar('s')
    _c = T.lscalar('c')

    # Define error function.
    g_hat = b_s[_s] + b_c[_c] + P[_s].T.dot(W).dot(f_sc)
    g = T.scalar('g')
    err = (g_hat - g) ** 2  # for squared loss

    # Define regularization function.
    norm = (fro_norm if args.regularization == 'fro' else
            l2_norm if args.regularization == 'l2' else
            l1_norm)
    reg = args.lambda_ * (norm(P[_s]) + norm(W))

    # The loss function minimizes least squares with regularization.
    loss = (err + reg).sum()  # the sum just pulls out the single value

    logging.info('compiling compute code')
    to_update = [b_s, b_c, P, W]
    gradients = dict(zip([v.name for v in to_update], T.grad(loss, to_update)))
    inputs = [f_sc, g, _s, _c]

    train_P = theano.function(
        inputs=inputs, outputs=loss, name='train_P',
        updates=[
            (P, T.set_subtensor(
                    P[_s],
                    T.maximum(
                        P_zeros,
                        P[_s] - args.lrate * gradients[P.name][_s])))
        ])
    train_W = theano.function(
        inputs=inputs, outputs=loss, name='train_W',
        updates=[
            (W, T.maximum(W_zeros, W - args.lrate * gradients[W.name]))
        ])
    train_B = theano.function(
        inputs=inputs, outputs=loss, name='train_B',
        updates=[
            (b_s,
             T.set_subtensor(
                  b_s[_s],
                  T.largest(
                      0, b_s[_s] - args.lrate * gradients[b_s.name][_s]))),
            (b_c,
             T.set_subtensor(
                  b_c[_c],
                  T.largest(
                      0, b_c[_c] - args.lrate * gradients[b_c.name][_c])))
        ])

    # predict = theano.function([f_sc, _s, _c], g_hat, name='predict')
    # def compute_rmse(Xframe, yframe, uids, iids):
    #     args_tuples = izip(Xframe.values, uids, iids)
    #     predictions = np.array([
    #         predict(*record) for record in args_tuples
    #     ]).reshape(len(Xframe))
    #     return rmse_from_err(predictions - yframe)

    _X = T.fmatrix('X')
    _uids = T.ivector('uids')
    _iids = T.ivector('iids')
    _y = T.fvector('y')

    _predicted, updates = theano.scan(
        fn=lambda x_i, _u, _i: b_s[_u] + b_c[_i] + P[_u].T.dot(W).dot(x_i),
        sequences=[_X, _uids, _iids])

    _err = _predicted.reshape(_y.shape) - _y
    compute_err = theano.function([_X, _y, _uids, _iids], outputs=_err,
                                   allow_input_downcast=True)

    _rmse = T.sqrt((_err ** 2).sum() / _err.shape[0])
    compute_rmse = theano.function(inputs=[_X, _y, _uids, _iids],
                                    outputs=_rmse, allow_input_downcast=True)

    def log_train_rmse():
        logging.info('Train RMSE:\t%.4f' % compute_rmse(
            train_x.values, train_y.values,
            train_uids.values, train_iids.values))

    def log_test_rmse():
        logging.info('Test RMSE:\t%.4f' % compute_rmse(
            test_x.values, test_y.values, test_uids.values, test_iids.values))

    # Main training loop.
    logging.info('training model for %d iterations' % args.iters)
    start = time.time()
    for it in range(args.iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (it + 1, elapsed))
        for train_model in [train_P, train_B, train_W]:
            logging.info('running %s' % train_model.name)
            index = np.random.permutation(train_x.index)
            training_set = izip(train_x.ix[index].values, train_y.ix[index],
                                train_uids[index], train_iids[index])

            for _args in training_set:
                train_model(*_args)

            if args.verbose == 2:
                log_train_rmse()
                log_test_rmse()

        if args.verbose == 1:
            log_train_rmse()
            log_test_rmse()

        log_shared()

    elapsed = time.time() - start
    logging.info('total time elapsed: %.2fs' % elapsed)

    logging.info('making predictions')
    print 'MLR RMSE:\t%.4f' % compute_rmse(
        test_x, test_y, test_uids, test_iids)

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(train_y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)
