# encoding: utf-8
# cython: profile=True

import time
import logging

import numpy as np

cimport cython
cimport numpy as np


def rmse_from_err(np.ndarray[np.double_t, ndim=1] err):
    return np.sqrt((err ** 2).sum() / err.shape[0])


def ipr_predict(dict model,
                object X,
                np.ndarray[np.long_t, ndim=1] eids,
                unsigned int nb):
    """Make predictions for each feature vector in X using the IPR model."""
    w0 = model['w0']
    w = model['w']
    P = model['P'][eids]
    W = model['W']

    B = X[:, :nb]
    X = X[:, nb:]
    return w0 + B.dot(w) + (X.dot(W.T) * P).sum(axis=1)


def compute_errors(dict model,
                   object X,
                   np.ndarray[np.double_t, ndim=1] y,
                   np.ndarray[np.long_t, ndim=1] eids,
                   unsigned int nb):
    return y - ipr_predict(model, X, eids, nb)


def compute_rmse(dict model,
                   object X,
                   np.ndarray[np.double_t, ndim=1] y,
                   np.ndarray[np.long_t, ndim=1] eids,
                   unsigned int nb):
    errors = compute_errors(model, X, y, eids, nb)
    return rmse_from_err(errors)


cdef np.ndarray randn(double std, tuple dim):
    return np.random.normal(0.0, std, dim)


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ipr_sgd(
        object X,
        np.ndarray[np.double_t, ndim=1] y,
        np.ndarray[np.long_t, ndim=1] eids,
        unsigned int nb,
        unsigned int k=3,
        double lrate=0.001,
        double lambda_w=0.01,
        double lambda_b=0.0,
        unsigned int iters=10,
        double eps=0.00001,
        double std=0.01,
        unsigned int nn=0,
        unsigned int verbose=0,
        object dtype=np.float64):

    cdef unsigned int b1, n, nf, p
    cdef np.ndarray[np.long_t, ndim=1] indices
    cdef np.ndarray[np.double_t, ndim=1] w, Pi_zeros
    cdef np.ndarray[np.double_t, ndim=2] P, W, W_zeros
    cdef double rmse, prev_rmse, start, elapsed, w0
    cdef dict model

    # Get value counts for parameter initialization.
    b1 = np.unique(eids).shape[0]  # num unique values for entity to profile
    n, nf = X.shape  # num training examples and num features
    p = nf - nb  # num non-entity predictor variables

    # Init params.
    w0 = 0
    w = randn(std, (nb,)).astype(dtype)
    P = randn(std, (b1, k)).astype(dtype)
    W = randn(std, (k, p)).astype(dtype)

    # init zero vectors for non-negative constrained optimization.
    Pi_zeros = np.zeros(k).astype(dtype)
    W_zeros = np.zeros((k, p)).astype(dtype)

    model = {
        'w0': w0,
        'w': w,
        'P': P,
        'W': W
    }

    # Declare vars for main opt loop.
    cdef unsigned int inum, t, i
    cdef double e_t, y_hat
    cdef np.ndarray[np.double_t, ndim=1] P_i
    cdef object X_, B_

    # Data setup. We assume bias terms are stored in the first nb columns.
    y = y.astype(dtype)
    X = X.tocsr().astype(dtype)
    B_ = X[:, :nb]  # n x nb
    X_ = X[:, nb:]  # n x p
    indices = np.arange(n)

    cdef np.ndarray[object, ndim=1] Brows, Xrows
    cdef np.ndarray[np.uint32_t, ndim=2] Bidx, Xidx, Bdata
    cdef np.ndarray[np.double_t, ndim=2] Xdata

    logging.info('warming up cache for sparse row indexing')
    logging.info('caching bias features')
    Brows = np.array([B_[i] for i in indices])
    Bidx = np.array([row.indices for row in Brows]).astype(np.uint32)
    Bdata = np.array([row.data for row in Brows]).astype(np.uint32)

    logging.info('caching non-bias features')
    Xrows = np.array([X_[i] for i in indices])
    Xidx = np.array([row.indices for row in Xrows]).astype(np.uint32)
    Xdata = np.array([row.data for row in Xrows])

    cdef object B_t
    cdef np.ndarray[np.double_t, ndim=1] reg, w_t, X_dat
    cdef np.ndarray[np.double_t, ndim=2] W_t
    cdef np.ndarray[np.uint32_t, ndim=1] B_dat, X_idx, B_idx

    # Compute initial error.
    rmse = compute_rmse(model, X, y, eids, nb)
    prev_rmse = np.inf
    logging.info('inital RMSE:\t%.4f' % rmse)

    # Main optimization loop.
    lrate *= 2  # fold in 2 to avoid recomputing it
    start = time.time()
    logging.info('training model for %d iterations' % iters)
    for inum in range(iters):
        elapsed = time.time() - start
        logging.info('iteration %03d\t(%.2fs)' % (inum + 1, elapsed))

        # Loop through all training examples.
        for t in np.random.permutation(indices):
            i = eids[t]
            P_i = P[i]
            B_t = Brows[t]

            B_dat = Bdata[t]
            B_idx = Bidx[t]
            X_dat = Xdata[t]
            X_idx = Xidx[t]

            w_t = w[B_idx]
            W_t = W[:, X_idx]

            reg = W_t.dot(X_dat)
            y_hat = w0 + B_dat.dot(w_t) + P_i.dot(reg)
            e_t = y_hat - y[t]
            if nn:
                w0 = np.maximum(0, w0 * lrate * e_t)
                w[B_idx] = np.maximum(
                    np.zeros(B_t.nnz), w_t - lrate * e_t * B_dat)
                P[i] = np.maximum(
                    Pi_zeros,
                    P_i - lrate * (e_t * reg + lambda_w * P_i))
                W = np.maximum(
                    W_zeros,
                    lrate * (
                        e_t * P_i[:, np.newaxis].dot(X_dat[np.newaxis, :]
                        + lambda_w * W_t)))
            else:
                w0 = w0 - lrate * e_t
                w[B_idx] = w_t - lrate * e_t * B_dat
                P[i] = P_i - lrate * (e_t * reg + lambda_w * P_i)
                W[:, X_idx] = W_t - lrate * (
                    e_t * P_i[:, np.newaxis].dot(X_dat[np.newaxis, :])
                    + lambda_w * W_t)

        model['w0'] = w0
        rmse = compute_rmse(model, X, y, eids, nb)
        logging.info('train RMSE:\t%.4f' % rmse)
        if prev_rmse - rmse < eps:
            logging.info('stopping threshold reached')
            break
        else:
            prev_rmse = rmse

    elapsed = time.time() - start
    logging.info('total time elapsed:\t(%.2fs)' % elapsed)

    return model
