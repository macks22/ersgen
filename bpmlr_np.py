"""
An implementation of the Bayesian formulation of the Personalized Multi-Linear
Regression (PMLR) model: Bayesian PMLR (BPMRL).

NON-PERSONALIZED VERSION.

NOTE: the gamma distribution in numpy uses the shape/scale parameterization. So
whenever we use np.random.gamma, we use shape=alpha, scale=(1/beta).

"""
import sys
import logging
import argparse

import scipy as sp
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import wishart
from scipy import stats

from util import load_np_vars


np.random.seed(1234)


# Error codes
BOUNDS_FORMAT = 1000
MISSING_ATTRIBUTE = 1001
DIM_MISMATCH = 1002


def cov(x):
    return np.cov(x, rowvar=0)

def solve(X):
    try:
        return np.linalg.cholesky(X)
    except np.linalg.LinAlgError:
        p, L, u = sp.linalg.lu(X)
        return L


def predict(alpha_G, p, W, X, bounds=(0, 4)):
    """Make predictions using BPMLR given parameters.
    Assumes features have already been scaled appropriately.
    """
    lo, hi = bounds

    means = p.reshape(1, L).dot(W).dot(X.T)[0]
    predictions = np.array([
        np.random.normal(means[i], alpha_G)
        for i in xrange(X.shape[0])
    ])
    predictions[predictions < lo] = lo
    predictions[predictions > hi] = hi
    return predictions


def make_parser():
    parser = argparse.ArgumentParser(
        description='BPMLR for non-cold-start dyadic response prediction')
    parser.add_argument(
        '-v', '--verbose',
        type=int, choices=(0, 1, 2), default=1,
        help='enable verbose logging output')
    parser.add_argument(
        '-tr', '--train',
        help='path of training data file in csv format')
    parser.add_argument(
        '-te', '--test',
        help='path of test data file in csv format')
    parser.add_argument(
        '-l', '--nmodels',
        type=int, default=3,
        help='number of regression models to use; 3 by default')
    parser.add_argument(
        '-u', '--uid', default='uid',
        help='user id column name in train/test files')
    parser.add_argument(
        '-i', '--iid', default='iid',
        help='item id column name in train/test files')
    parser.add_argument(
        '-t', '--target', default='target',
        help='target attribute column name in train/test files')
    parser.add_argument(
        '-ns', '--nsamples',
        type=int, default=50,
        help='number of samples to draw')
    parser.add_argument(
        '-b', '--burnin',
        type=int, default=5,
        help='number of burn-in samples')
    parser.add_argument(
        '-th', '--thin',
        type=int, default=2,
        help='how many samples to draw before keeping one, for thinning')
    parser.add_argument(
        '--bounds', default='0,4',
        help='upper,lower bound for rating bounding')
    parser.add_argument(
        '-s', '--stopping-threshold',
        type=float, default=0.00001,
        help='early stopping threshold')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose == 2 else
               logging.INFO if args.verbose == 1 else
               logging.ERROR),
        format="[%(asctime)s]: %(message)s")

    # Extract important column names from parser args namespace.
    uid, iid, target = args.uid, args.iid, args.target

    # Sanity check bounds argument.
    bounds = args.bounds.split(',')
    if len(bounds) != 2:
        print 'bounds must be comma-separated, got %s' % args.bounds
        sys.exit(BOUNDS_FORMAT)

    logging.info('reading train/test files')
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # Sanity check uid, iid, and target data keys.
    for key in [uid, iid, target]:
        for name, dataset in {'train': train, 'test': test}.items():
            if key not in dataset.columns:
                print 'key %s not in %s dataset' % (key, name)
                sys.exit(MISSING_ATTRIBUTE)

    # Split up training/test data.
    y = train[target].values
    X = train.drop([uid, iid, target], axis=1).values

    test_y = test[target].values
    test_X = test.drop([uid, iid, target], axis=1).values

    # Get various counts for logging.
    N = train[uid].unique().shape[0]
    M = train[iid].unique().shape[0]
    L = args.nmodels

    nd_train = train.shape[0]
    nd_test = test.shape[0]
    nd = nd_train + nd_test
    nf = X.shape[1]

    logging.info('N=%d, M=%d, L=%d' % (N, M, L))
    logging.info('# dyads: train=%d, test=%d, total=%d' % (
        nd_train, nd_test, nd))
    logging.info('%d features: %s' % (nf, ', '.join(train.columns)))

    # Z-score feature scaling
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    test_X = scaler.transform(test_X)

    # Init G precision.
    alpha_G = 2

    # Init p hyparparams.
    mu0_p = np.zeros(L)
    df0_p = L
    W0_p = np.eye(L)
    k0_p = 1

    # Init W hyperparams.
    mu0_W = np.zeros(nf)
    df0_W = nf
    W0_W = np.eye(nf)
    k0_W = 1

    # Init params using hyperparams.
    lambda_p = wishart.rvs(df0_p, W0_p)
    covar_p = np.linalg.inv(lambda_p * k0_p)
    mu_p = np.random.multivariate_normal(mu0_p, covar_p)

    lambda_W = wishart.rvs(df0_W, W0_W)
    covar_W = np.linalg.inv(lambda_W * k0_W)
    mu_W = np.random.multivariate_normal(mu0_W, covar_W)

    # Init vars using params.
    p = np.random.multivariate_normal(mu_p, covar_p)
    W = np.random.multivariate_normal(mu_W, covar_W, L)

    # Calculate initial predictions.
    overall_err = np.zeros(args.nsamples)
    nsamples = args.nsamples - args.burnin
    predictions = np.ndarray((nsamples + 1, nd_test))

    initial = predict(alpha_G, p, W, test_X, (0, 4))
    prev_rmse = np.sqrt(((initial - test_y) ** 2).sum() / test_y.shape[0])
    print 'initial RMSE: %.4f' % prev_rmse

    # Create space to store gibbs samples.
    dim = lambda mat: [args.nsamples + 1] + list(mat.shape)
    params = {}

    for var in ['p', 'W']:
        val = globals()[var]
        params[var] = np.ndarray(dim(val))
        params[var][0] = val

    for param in ['lambda_p', 'mu_p', 'lambda_W', 'mu_W']:
        val = globals()[param]
        params[param] = np.ndarray(dim(val))
        params[param][0] = val


    sample = 0
    def update(param):
        params[param][sample + 1] = globals()[param]


    # Begin Gibbs sampling.
    for sample in xrange(nsamples):
        print 'sample %d' % sample
        for sub_sample in xrange(args.thin):
            print 'sub-sample %d' % sub_sample

            # Update p hyperparams...
            k_post = k0_p + N
            p_bar = p.mean(axis=0)
            S_bar = cov(p)

            mu_tmp = mu0_p - p_bar
            W_post = np.linalg.inv(
                np.linalg.inv(W0_p) + (N * S_bar) +
                (k0_p * N) * mu_tmp.dot(mu_tmp.T) / k_post)
            W_post = (W_post + W_post.T) / 2  # Numerical correction?

            df_post = df0_p + N
            mu_post = (k0_p * mu0_p + N * p_bar) / k_post

            # ...and params.
            lambda_p = wishart.rvs(df_post, W_post)
            covar_p = np.linalg.inv(k_post * lambda_p)
            mu_p = np.random.multivariate_normal(mu_post, covar_p)

            update('lambda_p')
            update('mu_p')

            # Update W hyperparams...
            W_bar = W.mean(axis=0)
            S_bar = cov(W)
            k_post = k0_W + L

            mu_tmp = mu0_W - W_bar
            W_post = np.linalg.inv(
                np.linalg.inv(W0_W) + (L * S_bar) +
                (k0_W * L) * mu_tmp.dot(mu_tmp.T) / k_post)
            W_post = (W_post + W_post.T) / 2  # Numerical correction?

            df_post = df0_W + L
            mu_post = (k0_W * mu0_W + L * W_bar) / k_post

            # ...and params.
            lambda_W = wishart.rvs(df_post, W_post)
            covar_W = np.linalg.inv(k_post * lambda_W)
            mu_W = np.random.multivariate_normal(mu_post, covar_W)

            update('lambda_W')
            update('mu_W')

            # Update p for all users.
            lambda_pi = lambda_p + alpha_G * W.dot(X.T.dot(X)).dot(W.T)
            covar_pi = np.linalg.inv(lambda_pi)

            Xy = y.reshape(y.shape[0], 1) * X
            sum_term = W.dot(Xy.T).sum(axis=1)
            mu_pi = covar_pi.dot(lambda_p.dot(mu_p) + alpha_G * sum_term)
            p = np.random.multivariate_normal(mu_pi, covar_pi)

            # Update W
            X_sum = alpha_G * X.T.dot(X)
            Xy_sum = alpha_G * Xy.sum(axis=0)
            for l in xrange(L):
                p_l = p[l]
                lambda_Wl = lambda_W + (p_l ** 2) * X_sum
                covar_Wl = np.linalg.inv(lambda_Wl)
                mu_Wl = covar_Wl.dot(lambda_W.dot(mu_W) + p_l * Xy_sum)
                W[l] = np.random.multivariate_normal(mu_Wl, covar_Wl)


        # Store samples
        for var in ['p', 'W']:
            update(var)

        # After thinning, make predictions for each sample.
        predictions[sample] = predict(
            alpha_G, p, W, test_X, (0, 4))
        rmse = np.sqrt(((predictions - test_y) ** 2).sum() / test_y.shape[0])
        print 'rmse: %.4f' % rmse

        # Early stopping improvement threshold check.
        # if (prev_rmse - rmse) <= args.stopping_threshold:
        #     print '\tStopping threshold reached'
        #     break
        # else:
        #     prev_rmse = rmse
