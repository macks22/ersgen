"""
Implement Factorization Machine (FM) model.

The following learning algorithms have been implemented:

1.  Alternating Least Squares (ALS)

The command line expects data in CSV format, with separate files for training
and test records. A required file specifies the format of the attributes in the
CSV files. Three types of attributes are delineated: (1) target, (2)
categorical, and (3) real-valued. The target attribute is separated from the
others during training and predicted for the test data after the model is
learned. The categorical attributes are one-hot encoded, and the real-valued
attributes are scaled using Z-score scaling (0 mean, unit variance).

The format for the features file is as follows:

    t:<target>;
    c:<comma-separated categorical variable names>;
    r:<comma-separated real-valued variable names>;

Whitespace is ignored, as are lines that start with a "#" symbol. Any variables
not included in one of the three groups are ignored. They are used neither for
training nor prediction.

"""
import time
import logging
import itertools as it

import numpy as np
from scipy.linalg import blas

cimport cython
cimport numpy as np


def rmse_from_err(np.ndarray[np.double_t, ndim=1] err):
    return np.sqrt((err ** 2).sum() / err.shape[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def predict(X,
            double w0,
            np.ndarray[np.double_t, ndim=1] w,
            np.ndarray[np.double_t, ndim=2] V):
    """Predict y values for data X given model params w0, w, and V.

    Args:
        X (sp.sparse.coo.coo_matrix): Sparse data matrix with instances as rows.
        w0 (float): Global bias term.
        w (np.ndarray[np.double_t, ndim=1]): 1-way interaction terms.
        V (np.ndarray[np.double_t, ndim=2]): 2-way interaction terms.

    Returns:
        np.ndarray[np.double_t, ndim=1]: Predictions \hat{y}.

    """
    cdef unsigned int N = X.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] predictions = np.zeros(N) + w0

    cdef unsigned int f, i, j
    cdef double x_j, tmp
    cdef np.ndarray[np.double_t, ndim=1] t1, t2, v_f

    for f in xrange(V.shape[1]):
        t1 = np.zeros(N)
        t2 = np.zeros(N)
        v_f = V[:, f]
        for i, j, x_j in it.izip(X.row, X.col, X.data):
            tmp = v_f[j] * x_j
            t1[i] += tmp
            t2[i] += tmp ** 2

    predictions += ((t1 ** 2) - t2)  * 0.5

    for i, j, x in it.izip(X.row, X.col, X.data):
        predictions[i] += w[j] * x

    return predictions


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_fm_als(X,
               np.ndarray[np.double_t, ndim=1] y,
               unsigned int iters,
               double threshold,
               unsigned int k,
               double lambda_w,
               double lambda_v):

    # We have the data, let's begin.
    cdef int nd = X.shape[0]
    cdef int nf = X.shape[1]
    X_T = X.tocsc().T
    cdef np.ndarray[object, ndim=1] coldata, colrows
    for i in xrange(nf):
        coldata
    cdef np.ndarray[object, ndim=1] cols = np.array([
        X_T[i] for i in xrange(nf)
    ])
    cdef np.ndarray[object, ndim=1] coldata = np.array([
        col.data for col in cols
    ])
    cdef np.ndarray[object, ndim=1] colrows = np.array([
        col.indices for col in cols
    ])

    # Init w0, w, and V.
    cdef double w0 = 0
    cdef np.ndarray[np.double_t, ndim=1] w = np.zeros(nf)
    cdef np.ndarray[np.double_t, ndim=2] V = np.zeros((nf, k))

    # Precompute e and q.
    cdef np.ndarray[np.double_t, ndim=1] e
    e = predict(X, w0, w, V) - y

    cdef np.ndarray[np.double_t, ndim=2] q = np.zeros((nd, k))
    cdef unsigned int i, j, f
    cdef double x_j

    for f in xrange(k):
        for i, j, x_j in it.izip(X.row, X.col, X.data):
            q[i, f] += V[j, f] * x_j

    # Main optimization loop.
    cdef double rmse, prev_rmse, start
    prev_rmse = rmse_from_err(e)
    logging.info('initial RMSE: %.4f' % prev_rmse)
    start = time.time()

    # define all variables used in main loop.
    cdef double w0_new, w_new, v_jf, sum_nominator, sum_denominator, diff
    cdef np.ndarray[np.double_t, ndim=1] h, q_f, v_f, dat
    cdef np.ndarray[np.int32_t, ndim=1] rows
    cdef unsigned int iteration

    for iteration in xrange(iters):

        # Learn global bias term.
        w0_new = (e - w0).sum() / nd
        e += w0_new - w0
        w0 = w0_new

        # Learn 1-way interaction terms.
        for j in xrange(nf):
            dat = coldata[j]
            if not dat.shape[0]:
                w[j] = 0
                continue

            rows = colrows[j]
            sum_nominator = ((w[j] * dat - e[rows]) * dat).sum()
            sum_denominator = blas.ddot(dat, dat)
            w_new = sum_nominator / (sum_denominator + lambda_w)
            e[rows] += (w_new - w[j]) * dat
            w[j] = w_new

        # Learn 2-way interaction terms.
        for f in xrange(k):
            q_f = q[:, f]
            v_f = V[:, f]
            for j in xrange(nf):
                dat = coldata[j]
                if not dat.shape[0]:
                    V[j, f] = 0
                    continue

                rows = colrows[j]
                v_jf = v_f[j]

                h = dat * (q_f[rows] - dat * v_jf)
                sum_nominator = ((v_jf * h - e[rows]) * h).sum()
                sum_denominator = blas.ddot(h, h)

                v_new = sum_nominator / (sum_denominator + lambda_v)
                update = (v_new - v_jf) * dat
                e[rows]    += update
                q[rows, f] += update
                V[j, f] = v_new

        # Re-evaluate RMSE to prepare for stopping check.
        # Also recompute e every 100 iterations to correct gradual rounding err.
        if iteration % 100 == 0:
            e = predict(X, w0, w, V) - y

        rmse = rmse_from_err(e)
        logging.info(
            'RMSE after iteration %02d: %.4f  (%.2fs)' % (
                iteration, rmse, time.time() - start))

        if prev_rmse - rmse < threshold:
            logging.info('stopping threshold reached')
            break
        else:
            prev_rmse = rmse

    return w0, w, V

