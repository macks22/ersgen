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
        '-l', '--lambda',
        type=float, default=0.01,
        help='regularization multiplier')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.001,
        help='learning rate')
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

    train = pd.read_csv(args.train)
    train_y = train[args.target]
    train_x = train.drop(args.target, axis=1)
    test = pd.read_csv(args.test)
    test_y = test[args.target]
    test_x = test.drop(args.target, axis=1)

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
    b_s = theano.shared(value=np.random.randn(1), name='b_s', borrow=True)
    b_c = theano.shared(value=np.random.randn(1), name='b_c', borrow=True)
    p_s = theano.shared(value=np.zeros((l, 1)), name='p_s', borrow=True)
    f_sc = theano.shared(value=np.zeros((nf, 1)), name='f_sc')
    W = theano.shared(value=np.zeros((l, nf)), name='W', borrow=True)

    g_hat = b_s + b_c + p_s.T.dot(W).dot(f_sc)
    g = T.scalar('g')
    err = ((g_hat - g) ** 2) / 2  # not sure what this is at this point.
    reg = args.lambda * ((p_s ** 2).sum() + (W ** 2).sum())
    loss = (err + reg).sum()

    vars = [b_s, b_c, p_s, W, f_sc]
    gradients = dict(zip([v.name for v in vars], T.grad(loss, vars)))
    updates = [(v, v - args.lrate * gradients[v.name])
               for v in vars if v != f_sc]


