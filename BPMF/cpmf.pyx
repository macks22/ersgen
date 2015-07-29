"""
PMF Implementation in Python.

"""
import time
import logging

import numpy as np
cimport cython
cimport numpy as np

from util import predict, read_data


def fit_pmf(train,
            probe,
            str uid='uid',
            str iid='iid',
            str target='target',
            unsigned int epsilon=50,
            double lambda_=0.01,
            double momentum=0.8,
            unsigned int max_epoch=50,
            unsigned int nbatches=9,
            unsigned int N=100000,
            unsigned int nf=10):

    cdef double mean_rating = train[target].mean()
    cdef np.ndarray[np.double_t, ndim=1] ratings_test = \
        probe[target].values.astype(np.double)

    cdef unsigned int n = train[uid].max() + 1  # number of users
    cdef unsigned int m = train[iid].max() + 1  # number of items

    # Randomly initialize user and item feature vectors.
    cdef np.ndarray[np.double_t, ndim=2] w_u = 0.1 * np.random.randn(n, nf)  # User
    cdef np.ndarray[np.double_t, ndim=2] w_i = 0.1 * np.random.randn(m, nf)  # Item

    # Allocate space for feature vector update vectors.
    cdef np.ndarray[np.double_t, ndim=2] w_u_update = np.zeros((n, nf))
    cdef np.ndarray[np.double_t, ndim=2] w_i_update = np.zeros((m, nf))

    # Allocate space for error tracking vectors.
    cdef np.ndarray[np.double_t, ndim=1] err_train = np.zeros(max_epoch)
    cdef np.ndarray[np.double_t, ndim=1] err_valid = np.zeros(max_epoch)

    # Allocate other working variables.
    cdef np.ndarray[np.int_t, ndim=1] uids
    cdef np.ndarray[np.int_t, ndim=1] iids
    cdef np.ndarray[np.double_t, ndim=1] ratings, predictions, error, regular
    cdef np.ndarray[np.double_t, ndim=2] dw_u, dw_i, Ix_u, Ix_i, IO

    cdef unsigned int ii
    cdef unsigned int epoch, batch
    cdef np.ndarray[np.int_t, ndim=1] rr  ## rr = random range

    cdef double elapsed, f_s
    cdef double start = time.time()
    for epoch in xrange(max_epoch):
        rr = np.random.permutation(train.shape[0])
        train = train.ix[rr]

        for batch in xrange(nbatches):
            logging.debug('epoch %d, batch %d' % (epoch + 1, batch + 1))

            train_subset = train.ix[range(batch*N, (batch+1)*N)]
            uids = train_subset[uid].values
            iids = train_subset[iid].values
            ratings = train_subset[target].values.astype(np.double)

            # Default prediction is the mean rating, so subtract it.
            ratings = ratings - mean_rating

            # Compute predictions.
            predictions = np.sum(w_i[iids] * w_u[uids], 1)
            error = predictions - ratings
            # regular = np.sum(w_i[iids] ** 2 + w_u[uids] ** 2, 1)
            # loss = np.sum(error ** 2 + 0.5 * lambda_ * regular)

            # Compute gradients.
            IO = np.repeat(2 * error, nf).reshape(error.shape[0], nf)
            Ix_u = IO * w_i[iids] + lambda_ * w_u[uids]
            Ix_i = IO * w_u[uids] + lambda_ * w_i[iids]

            dw_u = np.zeros((n, nf))
            dw_i = np.zeros((m, nf))

            for ii in xrange(N):
                dw_u[uids[ii]] += Ix_u[ii]
                dw_i[iids[ii]] += Ix_i[ii]

            # Update user and item feature vectors.
            w_u_update = momentum * w_u_update + epsilon * (dw_u / N)
            w_u -= w_u_update

            w_i_update = momentum * w_i_update + epsilon * (dw_i / N)
            w_i -= w_i_update

        # Compute predictions after parameter updates.
        predictions = np.sum(w_i[iids] * w_u[uids], 1)
        error = predictions - ratings
        regular = np.sum(w_i[iids] ** 2 + w_u[uids] ** 2, 1)
        f_s = np.sum(error ** 2 + 0.5 * lambda_ * regular)
        err_train[epoch] = np.sqrt(f_s / N)

        # Compute predictions on the validation set.
        predictions = predict(w_u, w_i, mean_rating, probe)
        error = predictions - ratings_test

        elapsed = time.time() - start
        err_valid[epoch] = np.sqrt((error ** 2).sum() / probe.shape[0])
        logging.info(
            'epoch %2d, Training RMSE: %6.4f, Test RMSE: %6.4f (%.2fs)' % (
                epoch + 1, err_train[epoch], err_valid[epoch], elapsed))

        if (epoch + 1) % 10 == 0:
            np.savetxt('w_u.csv', w_u, delimiter=',')
            np.savetxt('w_i.csv', w_i, delimiter=',')

    return w_u, w_i
