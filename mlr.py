"""
Mixed-membership Multi-Linear Regression model implemented with Theano.

"""
import os
import sys
import logging
import argparse

import theano
import theano.tensor as T
import numpy as np
import pandas as pd


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
        action='store_true', default=False)
    parser.add_argument(
        '-o', '--output',
        action='store_true', default=False)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
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
    n = len(train_x[uid].unique())
    m = len(train_x[iid].unique())
    l = args.nmodels

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
    randn = lambda dim: np.random.normal(0, 0.1, dim)
    b_s = theano.shared(value=randn(1), name='b_s', borrow=True)
    b_c = theano.shared(value=randn(1), name='b_c', borrow=True)
    p_s = theano.shared(value=randn((l, 1)), name='p_s', borrow=True)
    W = theano.shared(value=randn((l, nf)), name='W', borrow=True)
    f_sc = T.dvector('f_sc')  # (nf, 1)

    print '%s: %.4f' % (b_s.name, b_s.get_value()[0])
    print '%s: %.4f' % (b_c.name, b_c.get_value()[0])
    print '%s:\n%s' % (p_s.name, str(p_s.get_value()))
    print '%s:\n%s' % (W.name, str(W.get_value()))

    g_hat = b_s + b_c + p_s.T.dot(W).dot(f_sc)
    g = T.scalar('g')
    err = 0.5 * ((g_hat - g) ** 2)  # not sure if this is correct for RMSE
    reg = args.lambda_ * ((p_s ** 2).sum() + (W ** 2).sum())
    loss = (err + reg).sum()  # the sum just pulls out the single value

    to_update = [b_s, b_c, W, p_s]
    gradients = dict(zip([v.name for v in to_update], T.grad(loss, to_update)))
    updates = [(v, v - args.lrate * gradients[v.name]) for v in to_update]
    # updates[-1] = (p_s, (p_s - args.lrate * gradients[p_s.name]) / p_s.sum())
    train_P = theano.function(
        inputs=[f_sc, g], outputs=loss, name='train_P',
        updates=[(p_s, (p_s - args.lrate * gradients[p_s.name]) / p_s.sum())])
    train_B = theano.function(
        inputs=[f_sc, g], outputs=loss, name='train_B',
        updates=[(b_s, b_s - args.lrate * gradients[b_s.name]),
                 (b_c, b_c - args.lrate * gradients[b_c.name])])
    train_W = theano.function(
        inputs=[f_sc, g], outputs=loss, name='train_W',
        updates=[(W, W - args.lrate * gradients[W.name])])

    for it in range(args.iters):
        for train_model in [train_P, train_B, train_W]:
            for idx in np.random.permutation(train_x.index):
                train_model(train_x.ix[idx], train_y.ix[idx])

        print '%s: %.4f' % (b_s.name, b_s.get_value()[0])
        print '%s: %.4f' % (b_c.name, b_c.get_value()[0])
        print '%s:\n%s' % (p_s.name, str(p_s.get_value()))
        print '%s:\n%s' % (W.name, str(W.get_value()))

    predict = theano.function([f_sc], g_hat)
    predictions = np.array([predict(test_x.ix[idx]) for idx in test_x.index])
    predictions = predictions.reshape(len(predictions))  # make 1D
    err = predictions - test_y
    rmse = np.sqrt((err ** 2).sum() / len(err))
    print 'RMSE:\t%.4f' % rmse

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    err = baseline_pred - test_y
    rmse = np.sqrt((err ** 2).sum() / len(err))
    print 'baseline RMSE:\t%.4f' % rmse
