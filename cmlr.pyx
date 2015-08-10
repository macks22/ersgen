"""
Mixed-membership Multi-Linear Regression model (non-Theano version).
"""
import time
import logging

import numpy as np
import pandas as pd

cimport cython
cimport numpy as np

ctypedef np.int_t INT_t
ctypedef np.double_t DOUBLE_t


def rmse_from_err(np.ndarray err):
    return np.sqrt((err ** 2).sum() / err.shape[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def mlr_predict(dict model,
                np.ndarray[DOUBLE_t, ndim=2] X,
                np.ndarray[INT_t, ndim=1] uids,
                np.ndarray[INT_t, ndim=1] iids):

    cdef np.ndarray[DOUBLE_t, ndim=1] b_s = model['b_s'][uids]
    cdef np.ndarray[DOUBLE_t, ndim=1] b_c = model['b_c'][iids]
    cdef np.ndarray[DOUBLE_t, ndim=3] P = model['P'][uids]
    cdef np.ndarray[DOUBLE_t, ndim=2] W = model['W']

    cdef unsigned int i
    return np.array([
        b_s[i] + b_c[i] + P[i].T.dot(W).dot(X[i])
        for i in xrange(X.shape[0])
    ]).reshape(X.shape[0])


def compute_errors(dict model,
                   np.ndarray[DOUBLE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=1] y,
                   np.ndarray[INT_t, ndim=1] uids,
                   np.ndarray[INT_t, ndim=1] iids):

    cdef np.ndarray[DOUBLE_t, ndim=1] predictions = \
        mlr_predict(model, X, uids, iids)
    return predictions - y


def compute_rmse(dict model,
                 np.ndarray[DOUBLE_t, ndim=2] X,
                 np.ndarray[DOUBLE_t, ndim=1] y,
                 np.ndarray[INT_t, ndim=1] uids,
                 np.ndarray[INT_t, ndim=1] iids):
    cdef np.ndarray[DOUBLE_t, ndim=1] errors = \
        compute_errors(model, X, y, uids, iids)
    return rmse_from_err(errors)


cdef np.ndarray randn(double std, tuple dim):
    return np.random.normal(0.01, std, dim)


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_mlr(np.ndarray[DOUBLE_t, ndim=2] _X,
            np.ndarray[DOUBLE_t, ndim=1] y,
            np.ndarray[INT_t, ndim=1] uids,
            np.ndarray[INT_t, ndim=1] iids,
            unsigned int l=3,
            double lrate=0.001,
            double lambda_=0.01,
            unsigned int iters=10,
            double std=0.01,
            unsigned int verbose=0):

    cdef unsigned int n, m, nd, nf
    n = np.unique(uids).shape[0]
    m = np.unique(iids).shape[0]
    nd, nf = _X.shape[:2]
    cdef np.ndarray[DOUBLE_t, ndim=3] X = _X.reshape((nd, nf, 1))

    #randn = lambda dim: np.random.normal(0.01, std, dim)
    cdef np.ndarray[DOUBLE_t, ndim=1] b_s = randn(std, (n,))
    cdef np.ndarray[DOUBLE_t, ndim=1] b_c = randn(std, (m,))
    cdef np.ndarray[DOUBLE_t, ndim=3] P = randn(std, (n, l, 1))
    cdef np.ndarray[DOUBLE_t, ndim=2] W = randn(std, (l, nf))

    model = {
        'b_s': b_s,
        'b_c': b_c,
        'P': P,
        'W': W
    }

    cdef np.ndarray[INT_t, ndim=1] indices = np.arange(nd)
    cdef double start, elapsed
    cdef double y_hat, error
    cdef unsigned int _sc, _iter, s, c

    logging.info('training model for %d iterations' % iters)
    start = time.time()
    for _iter in range(iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (_iter + 1, elapsed))
        for _sc in np.random.permutation(indices):
            s = uids[_sc]
            c = iids[_sc]
            P_s = P[s]

            # compute error
            y_hat = (b_s[s] + b_c[c] + P_s.T.dot(W).dot(X[_sc]))
            error = lrate * 2 * (y_hat - y[_sc])

            # update parameters
            P[s]   -= error * W.dot(X[_sc]) + 2 * lambda_ * P_s
            b_s[s] -= error
            b_c[c] -= error
            W      -= error * P_s.dot(X[_sc].T) + 2 * lambda_ * W

        #TODO: if early stopping is implemented, remove conditional.
        if verbose >= 1:  # conditional to avoid unnecessary computation
            logging.info('Train RMSE:\t%.4f' % compute_rmse(
                model, _X, y, uids, iids))

    elapsed = time.time() - start
    logging.info('total time elapsed:\t(%.2fs)' % elapsed)

    return model
