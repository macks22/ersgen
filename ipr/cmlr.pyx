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

    cdef int w0 = model['w0']
    cdef np.ndarray[DOUBLE_t, ndim=1] s = model['s'][uids]
    cdef np.ndarray[DOUBLE_t, ndim=1] c = model['c'][iids]
    cdef np.ndarray[DOUBLE_t, ndim=3] P = model['P'][uids]
    cdef np.ndarray[DOUBLE_t, ndim=2] W = model['W']

    cdef unsigned int i
    return np.array([
        s[i] + c[i] + P[i].T.dot(W).dot(X[i]) + w0
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
def fit_mlr_sgd(
        np.ndarray[DOUBLE_t, ndim=2] _X,
        np.ndarray[DOUBLE_t, ndim=1] y,
        np.ndarray[INT_t, ndim=1] uids,
        np.ndarray[INT_t, ndim=1] iids,
        unsigned int l=3,
        double lrate=0.001,
        double lambda_w=0.01,
        double lambda_b=0.001,
        unsigned int iters=10,
        double std=0.01,
        unsigned int verbose=0):

    cdef unsigned int n, m, nd, nf
    n = np.unique(uids).shape[0]
    m = np.unique(iids).shape[0]
    nd, nf = _X.shape[:2]
    cdef np.ndarray[DOUBLE_t, ndim=3] X = _X.reshape((nd, nf, 1))

    cdef int w0 = np.mean(y)
    cdef np.ndarray[DOUBLE_t, ndim=1] s = randn(std, (n,))
    cdef np.ndarray[DOUBLE_t, ndim=1] c = randn(std, (m,))
    cdef np.ndarray[DOUBLE_t, ndim=3] P = randn(std, (n, l, 1))
    cdef np.ndarray[DOUBLE_t, ndim=2] W = randn(std, (l, nf))

    model = {
        'w0': w0,
        's': s,
        'c': c,
        'P': P,
        'W': W
    }

    cdef np.ndarray[INT_t, ndim=1] indices = np.arange(nd)
    cdef double start, elapsed
    cdef double y_hat, error
    cdef unsigned int _sc, _iter, _s, _c
    cdef double learn = 2 * lrate

    logging.info('training model for %d iterations' % iters)
    start = time.time()
    for _iter in range(iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (_iter + 1, elapsed))
        for _sc in np.random.permutation(indices):
            _s = uids[_sc]
            _c = iids[_sc]
            P_s = P[_s]

            # compute error
            y_hat = s[_s] + c[_c] + P_s.T.dot(W).dot(X[_sc]) + w0
            error = (y_hat - y[_sc])

            # update parameters
            P[_s] -= learn * (
                error * W.dot(X[_sc]) + lambda_w * P_s)
            s[_s] -= learn * (error + lambda_b * s[_s])
            c[_c] -= learn * (error + 2 * lambda_b * c[_c])
            W -= learn * (
                error * P_s.dot(X[_sc].T) + lambda_w * W)

        #TODO: if early stopping is implemented, remove conditional.
        if verbose >= 1:  # conditional to avoid unnecessary computation
            logging.info('Train RMSE:\t%.4f' % compute_rmse(
                model, _X, y, uids, iids))

    elapsed = time.time() - start
    logging.info('total time elapsed:\t(%.2fs)' % elapsed)

    return model


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_mlr_sgd_nn(
        np.ndarray[DOUBLE_t, ndim=2] _X,
        np.ndarray[DOUBLE_t, ndim=1] y,
        np.ndarray[INT_t, ndim=1] uids,
        np.ndarray[INT_t, ndim=1] iids,
        unsigned int l=3,
        double lrate=0.001,
        double lambda_w=0.01,
        double lambda_b=0.001,
        unsigned int iters=10,
        double std=0.01,
        unsigned int verbose=0):

    cdef unsigned int n, m, nd, nf
    n = np.unique(uids).shape[0]
    m = np.unique(iids).shape[0]
    nd, nf = _X.shape[:2]
    cdef np.ndarray[DOUBLE_t, ndim=3] X = _X.reshape((nd, nf, 1))

    cdef int w0 = np.mean(y)
    cdef np.ndarray[DOUBLE_t, ndim=1] s = randn(std, (n,))
    cdef np.ndarray[DOUBLE_t, ndim=1] c = randn(std, (m,))
    cdef np.ndarray[DOUBLE_t, ndim=3] P = randn(std, (n, l, 1))
    cdef np.ndarray[DOUBLE_t, ndim=2] W = randn(std, (l, nf))

    model = {
        'w0': w0,
        's': s,
        'c': c,
        'P': P,
        'W': W
    }

    cdef np.ndarray[INT_t, ndim=1] indices = np.arange(nd)
    cdef double start, elapsed
    cdef double y_hat, error
    cdef unsigned int _sc, _iter, _s, _c
    cdef double learn = 2 * lrate

    cdef np.ndarray[DOUBLE_t, ndim=2] P_zeros = np.zeros((l, 1))
    cdef np.ndarray[DOUBLE_t, ndim=2] W_zeros = np.zeros((l, nf))

    logging.info('training model for %d iterations' % iters)
    start = time.time()
    for _iter in range(iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (_iter + 1, elapsed))
        for _sc in np.random.permutation(indices):
            _s = uids[_sc]
            _c = iids[_sc]
            P_s = P[_s]

            # compute error
            y_hat = s[_s] + c[_c] + P_s.T.dot(W).dot(X[_sc]) + w0
            error = (y_hat - y[_sc])

            # update parameters
            P[_s] = np.maximum(
                P_zeros,
                P[_s] - learn * (error * W.dot(X[_sc]) + lambda_w * P_s))
            s[_s] = np.maximum(
                0,
                s[_s] - learn * (error + lambda_b * s[_s]))
            c[_c] = np.maximum(
                0,
                s[_c] - learn * (error + lambda_b * c[_c]))
            W = np.maximum(
                W_zeros,
                W - learn * (error * P_s.dot(X[_sc].T) + lambda_w * W))

        #TODO: if early stopping is implemented, remove conditional.
        if verbose >= 1:  # conditional to avoid unnecessary computation
            logging.info('Train RMSE:\t%.4f' % compute_rmse(
                model, _X, y, uids, iids))

    elapsed = time.time() - start
    logging.info('total time elapsed:\t(%.2fs)' % elapsed)

    return model


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_mlr_als(
        np.ndarray[DOUBLE_t, ndim=2] _X,
        np.ndarray[DOUBLE_t, ndim=1] y,
        np.ndarray[INT_t, ndim=1] uids,
        np.ndarray[INT_t, ndim=1] iids,
        unsigned int l=3,
        double lambda_w=0.01,
        double lambda_b=0.001,
        unsigned int iters=10,
        double std=0.01,
        unsigned int verbose=0):

    cdef unsigned int n, m, nd, nf
    n = np.unique(uids).shape[0]
    m = np.unique(iids).shape[0]
    nd, nf = _X.shape[:2]
    cdef np.ndarray[DOUBLE_t, ndim=3] X = _X.reshape((nd, nf, 1))

    # Init params.
    cdef int w0 = np.mean(y)
    cdef np.ndarray[DOUBLE_t, ndim=1] s = np.zeros(n)
    cdef np.ndarray[DOUBLE_t, ndim=1] c = np.zeros(m)
    cdef np.ndarray[DOUBLE_t, ndim=3] P = randn(std, (n, l, 1))
    cdef np.ndarray[DOUBLE_t, ndim=2] W = randn(std, (l, nf))
    cdef np.ndarray[DOUBLE_t, ndim=2] ones = np.ones((l, nf))

    # Subtract mean from target variable.
    cdef np.ndarray[DOUBLE_t, ndim=1] y_cent = y - w0

    model = {
        'w0': w0,
        's': s,
        'c': c,
        'P': P,
        'W': W
    }

    cdef np.ndarray[np.uint32_t, ndim=1] indices = np.arange(nd).astype(np.uint32)
    cdef np.ndarray[DOUBLE_t, ndim=1] cache1d_1, cache1d_2, X_sc
    cdef double start, elapsed, tmp, deriv, cache, y_sc
    cdef unsigned int _iter, _sc, _s, _c, _l, f

    logging.info('training model for %d iterations' % iters)
    logging.info('initial RMSE:\t%.4f' % compute_rmse(
        model, _X, y_cent, uids, iids))

    start = time.time()
    for _iter in range(iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (_iter + 1, elapsed))

        for _sc in np.random.permutation(indices):
            _s = uids[_sc]
            _c = iids[_sc]
            y_sc = y_cent[_sc]
            X_sc = _X[_sc]
            cache = P[_s].T.dot(W).dot(X_sc)

            # first learn the student bias terms.
            s[_s] = y_sc - c[_c] - cache

            # next update the course bias terms.
            c[_c] = y_sc - s[_s] - cache

            # finally, update regression coefficients W
            for _l in xrange(l):
                for f in xrange(nf):
                    deriv = P[_s][_l] * X_sc[f]
                    tmp = P[_s].T.dot(W).dot(X_sc) - P[_s][_l] * W[_l,f] * X_sc[f]
                    tmp = cache - tmp
                    W[_l,f] = (deriv * tmp) / (lambda_w + deriv ** 2)

            # next update student memberships P.
            cache1d_1 = W.dot(X_sc)
            cache = y_sc - s[_s] - c[_c]
            for _l in xrange(l):
                deriv = cache1d_1[l]
                cache1d_2 = P[_s].squeeze() * cache1d_1
                tmp = cache - (cache1d_2[:_l].sum() + cache1d_2[_l+1:].sum())
                P[_s][_l] = (deriv * tmp) / (lambda_w + deriv ** 2)

        #TODO: if early stopping is implemented, remove conditional.
        if verbose >= 1:  # conditional to avoid unnecessary computation
            logging.info('Train RMSE:\t%.4f' % compute_rmse(
                model, _X, y_cent, uids, iids))

    elapsed = time.time() - start
    logging.info('total time elapsed:\t(%.2fs)' % elapsed)

    return model
