"""
Mixed-membership Multi-Linear Regression model implemented with Theano.

"""
import os
import sys
import time
import logging
import argparse

import theano
import theano.tensor as T
import numpy as np
import pandas as pd


def compute_rmse(err):
    return np.sqrt((err ** 2).sum() / len(err))


def make_parser():
    parser = argparse.ArgumentParser(
        description="mixed-membership multi-linear regression")
    parser.add_argument(
        '-tr', '--train',
        help='path of training data file')
    parser.add_argument(
        '-te', '--test',
        help='path of test data file')
    parser.add_argument(
        '-n', '--nmodels',
        type=int, default=3,
        help='number of linear regression models')
    parser.add_argument(
        '-l', '--lambda_',
        type=float, default=0.01,
        help='regularization multiplier')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-i', '--iters',
        type=int, default=10,
        help='number of iterations')
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

    logging.info('reading train/test data')
    data = pd.read_csv('data-n500-m50-t2-d3054.csv')
    prediction_term = data.term.max()
    test = data[data.term == prediction_term]
    train = data[data.term < prediction_term]

    logging.info('splitting train/test sets into X, y')
    train_y = train[args.target]
    train_x = train.drop(args.target, axis=1)
    test_y = test[args.target]
    test_x = test.drop(args.target, axis=1)

    # Get dimensions of training data.
    nd, nf = train_x.shape
    l = args.nmodels

    # Map user ids to bias indices.
    uids = data[uid].unique()
    n = len(uids)
    uid_map = dict(zip(uids, range(n)))

    # Map item ids to bias indices.
    iids = data[iid].unique()
    m = len(iids)
    iid_map = dict(zip(iids, range(m)))

    logging.info('%d users, %d items' % (n, m))
    logging.info('%d dyads with %d features' % (nd, nf))
    logging.info('l=%d, lr=%f' % (l, args.lrate))

    # train = pd.read_csv(args.train)
    # train_y = train[args.target]
    # train_x = train.drop(args.target, axis=1)
    # test = pd.read_csv(args.test)
    # test_y = test[args.target]
    # test_x = test.drop(args.target, axis=1)

    # Set up Theano variables for single-instance incremental learning.
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

    _s = T.lscalar('s')
    _c = T.lscalar('c')

    g_hat = b_s[_s] + b_c[_c] + P[_s].T.dot(W).dot(f_sc)
    g = T.scalar('g')
    err = 0.5 * ((g_hat - g) ** 2)  # not sure if this is correct for RMSE
    reg = args.lambda_ * ((P[_s] ** 2).sum() + (W ** 2).sum())
    loss = (err + reg).sum()  # the sum just pulls out the single value

    logging.info('compiling compute code')
    to_update = [b_s, b_c, P, W]
    gradients = dict(zip([v.name for v in to_update], T.grad(loss, to_update)))
    updates = [(v, v - args.lrate * gradients[v.name]) for v in to_update]
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

    # Main training loop.
    logging.info('training model for %d iterations' % args.iters)
    start = time.time()
    for it in range(args.iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (it + 1, elapsed))
        for train_model in [train_P, train_B, train_W]:
            for idx in np.random.permutation(train_x.index):
                x = train_x.ix[idx]
                train_model(x, train_y.ix[idx],
                            uid_map[x['sid']], iid_map[x['cid']])

        log_shared()

    elapsed = time.time() - start
    logging.info('total time elapsed: %.2fs' % elapsed)

    logging.info('making predictions')
    predict = theano.function([f_sc, _s, _c], g_hat)
    predictions = np.array([
        predict(test_x.ix[idx],
                uid_map[test_x.ix[idx]['sid']],
                iid_map[test_x.ix[idx]['cid']])
        for idx in test_x.index
    ]).reshape(len(test_x))  # make 1D

    err = predictions - test_y
    rmse = compute_rmse(err)
    print 'MLR RMSE:\t%.4f' % rmse

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    err = baseline_pred - test_y
    rmse = compute_rmse(err)
    print 'baseline RMSE:\t%.4f' % rmse
