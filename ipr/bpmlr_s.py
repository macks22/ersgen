"""
An implementation of the Bayesian formulation of the Personalized Multi-Linear
Regression (PMLR) model: Bayesian PMLR (BPMRL).

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


def predict(alpha_G, s, c, P, W, X, uids, iids):
    """Make predictions using BPMLR given parameters.
    Assumes features have already been scaled appropriately.
    """
    sbias = s[uids]
    cbias = c[iids]
    membs = P[uids]

    means = sbias + cbias + np.sum(membs.dot(W) * X, 1)
    sigma_G = np.sqrt(1.0 / alpha_G)
    predictions = np.array([
        np.random.normal(means[i], sigma_G)
        for i in xrange(X.shape[0])
    ])
    return predictions


def map_ids(data, key, id_map=None):
    """Map ids to 0-contiguous index. This enables the use of these ids as
    indices into an array (for the bias terms, for instance). This returns the
    number of unique IDs for `key`.
    """
    if id_map is None:
        ids = data[key].unique()
        n = len(ids)
        id_map = dict(zip(ids, range(n)))
    else:
        data[key] = data[key].apply(lambda _id: id_map[_id])
    return id_map


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
    parser.add_argument(
        '-w', '--warm-start',
        default='',
        help='init (warm-start) params from saved values in this directory')
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

    # Map uid and iid to 0-contiguous indices
    uid_map = map_ids(train, uid)
    iid_map = map_ids(train, iid)
    map_ids(test, uid, uid_map)
    map_ids(test, iid, iid_map)

    # Split up training/test data.
    uids = train[uid].values
    iids = train[iid].values
    y = train[target].values
    X = train.drop([uid, iid, target], axis=1).values

    test_uids = test[uid].values
    test_iids = test[iid].values
    test_y = test[target].values
    test_X = test.drop([uid, iid, target], axis=1).values

    uniq_uids = np.unique(uids)
    uniq_iids = np.unique(iids)

    # Get various counts for logging.
    N = uniq_uids.shape[0]
    M = uniq_iids.shape[0]
    L = args.nmodels

    nd_train = train.shape[0]
    nd_test = test.shape[0]
    nd = nd_train + nd_test
    nf = train.columns.shape[0] - 3  # ignore uid, iid, target

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

    # Init s hyperparams.
    mu0_s = 0
    alpha0_s = 1
    beta0_s = 1
    k0_s = 1

    # Init c hyperparams.
    mu0_c = 0
    alpha0_c = 1
    beta0_c = 1
    k0_c = 1

    # Init P hyparparams.
    mu0_P = np.zeros(L)
    alpha_P = np.eye(L) * 2
    df0_P = L
    W0_P = np.eye(L)
    k0_P = 1

    # Init W hyperparams.
    # mu0_W = np.zeros(nf)
    mu0_W = np.ones(nf)
    alpha_W = np.eye(nf) * 2
    df0_W = nf
    W0_W = np.eye(nf)
    k0_W = 1

    # Finally, init actual model variables...
    if args.warm_start:  # ...from saved values.
        model = load_np_vars(args.warm_start)
        s = model['s']
        c = model['c']
        P = model['P']
        W = model['W']

        # Convert P to 2-dimensional.
        P = P.reshape(P.shape[0], P.shape[1])

        # Sanity check dimensions.
        if s.shape[0] != N:
            print 'Dimension 1 of s: %d != %d' % (s.shape[0], N)
            sys.exit(DIM_MISMATCH)
        elif c.shape[0] != M:
            print 'Dimension 1 of c: %d != %d' % (c.shape[0], M)
            sys.exit(DIM_MISMATCH)
        elif P.shape[0] != N:
            print 'Dimension 1 of P: %d != %d' % (P.shape[0], N)
            sys.exit(DIM_MISMATCH)
        elif P.shape[1] != L:
            print 'Dimension 2 of P: %d != %d' % (P.shape[1], L)
            sys.exit(DIM_MISMATCH)
        elif W.shape[0] != L:
            print 'Dimension 1 of W: %d != %d' % (W.shape[0], L)
            sys.exit(DIM_MISMATCH)
        elif W.shape[1] != nf:
            print 'Dimension 2 of W: %d != %d' % (W.shape[1], nf)
            sys.exit(DIM_MISMATCH)

        # Init params from saved random vars.
        mu_s = s.mean()
        alpha_s = 1 / s.var(ddof=1)

        mu_c = c.mean()
        alpha_c = 1 / c.var(ddof=1)

        mu_P = P.mean(axis=0)
        lambda_P = np.linalg.inv(cov(P))

        mu_W = W.mean(axis=0)
        lambda_W = np.linalg.inv(cov(W))

    else:  # ...randomly using hyperparameters.
        # Init params using hyperparams.
        alpha_s = np.random.gamma(alpha0_s, 1 / beta0_s)
        sigma_s = np.sqrt(1 / (k0_s * alpha_s))
        mu_s = np.random.normal(mu0_s, sigma_s)

        alpha_c = np.random.gamma(alpha0_c, 1 / beta0_c)
        sigma_c = np.sqrt(1 / (k0_c * alpha_c))
        mu_c = np.random.normal(mu0_c, sigma_c)

        # alpha_P = np.diag(wishart.rvs(df0_P, W0_P))
        lambda_P = np.eye(L) * alpha_P
        covar_P = np.linalg.inv(lambda_P * k0_P)
        mu_P = np.random.multivariate_normal(mu0_P, covar_P)

        alpha_W = np.diag(wishart.rvs(df0_W, W0_W))
        lambda_W = np.eye(nf) * alpha_W
        covar_W = np.linalg.inv(lambda_W * k0_W)
        mu_W = np.random.multivariate_normal(mu0_W, covar_W)

        # Init vars using params.
        s = np.random.normal(mu_s, np.sqrt(1 / alpha_s), N)
        c = np.random.normal(mu_c, np.sqrt(1 / alpha_c), M)
        P = np.random.multivariate_normal(mu_P, covar_P, N)
        W = np.random.multivariate_normal(mu_W, covar_W, L)


    # Calculate initial predictions.
    overall_err = np.zeros(args.nsamples)
    nsamples = args.nsamples - args.burnin
    predictions = np.ndarray((nsamples + 1, nd_train))

    initial = predict(alpha_G, s, c, P, W, X, uids, iids)
    prev_rmse = np.sqrt(((initial - y) ** 2).sum() / y.shape[0])
    logging.info('initial RMSE: %.4f' % prev_rmse)

    # Create space to store gibbs samples.
    dim = lambda mat: [args.nsamples + 1] + list(mat.shape)
    params = {}

    for var in ['s', 'c', 'P', 'W']:
        val = globals()[var]
        params[var] = np.ndarray(dim(val))
        params[var][0] = val

    for param in ['mu_s', 'alpha_s', 'mu_c', 'alpha_c']:
        params[param] = np.ndarray(args.nsamples + 1)
        params[param][0] = globals()[param]

    for param in ['lambda_P', 'mu_P', 'lambda_W', 'mu_W']:
        val = globals()[param]
        params[param] = np.ndarray(dim(val))
        params[param][0] = val

    sample = 0
    def update(param):
        params[param][sample + 1] = globals()[param]


    # Begin Gibbs sampling.
    for sample in xrange(nsamples):
        logging.info('sample %d' % sample)
        for sub_sample in xrange(args.thin):

            # Sample from s hyperparams...
            s_mean = s.mean()
            k_post = k0_s + N
            beta0_post = (beta0_s +
                0.5 * ((N * s.var()) +
                       (N * k0_s) * ((s_mean - mu0_s) ** 2) / k_post))

            # mu0_post = (k0_s * mu0_s + N * s_mean) / k_post
            alpha0_post = alpha0_s + (N / 2)

            # ...and params.
            alpha_s = np.random.gamma(alpha0_post, 1 / beta0_post)
            sigma_s = np.sqrt(1 / (k0_s * alpha_s))
            # mu_s = np.random.normal(mu0_post, sigma_s)
            mu_s = np.random.normal(mu0_s, sigma_s)

            update('alpha_s')
            update('mu_s')

            # Update c hyperparams...
            c_mean = c.mean()
            k_post = k0_c + M
            beta0_post = (beta0_c +
                0.5 * ((M * c.var()) +
                       (M * k0_c) * ((c_mean - mu0_c) ** 2) / k_post))

            # mu0_post = (k0_c * mu0_c + M * c_mean) / k_post
            alpha0_post = alpha0_c + (M / 2)

            # ...and params.
            alpha_c = np.random.gamma(alpha0_post, 1 / beta0_post)
            sigma_c = np.sqrt(1 / (k0_c * alpha_c))
            # mu_c = np.random.normal(mu0_post, sigma_c)
            mu_c = np.random.normal(mu0_c, sigma_c)

            update('alpha_c')
            update('mu_c')

            # Update P hyperparams...
            k_post = k0_P + N
            P_bar = P.mean(axis=0)
            S_bar = cov(P)

            mu_tmp = mu0_P - P_bar
            W_post = np.linalg.inv(
                np.linalg.inv(W0_P) + (N * S_bar) +
                (k0_P * N) * mu_tmp.dot(mu_tmp.T) / k_post)

            df_post = df0_P + N
            # mu_post = (k0_P * mu0_P + N * P_bar) / k_post

            # ...and params.
            # alpha_P = np.diag(wishart.rvs(df_post, W_post))
            lambda_P = np.eye(L) * alpha_P
            covar_P = np.linalg.inv(k_post * lambda_P)
            # mu_P = np.random.multivariate_normal(mu_post, covar_P)
            mu_P = np.random.multivariate_normal(mu0_P, covar_P)

            update('lambda_P')
            update('mu_P')

            # Update W hyperparams...
            W_bar = W.mean(axis=0)
            S_bar = cov(W)
            k_post = k0_W + L

            mu_tmp = mu0_W - W_bar
            W_post = np.linalg.inv(
                np.linalg.inv(W0_W) + (L * S_bar) +
                (k0_W * L) * mu_tmp.dot(mu_tmp.T) / k_post)

            df_post = df0_W + L
            # mu_post = (k0_W * mu0_W + L * W_bar) / k_post

            # ...and params.
            alpha_W = np.diag(wishart.rvs(df_post, W_post))
            lambda_W = np.eye(nf) * alpha_W
            covar_W = np.linalg.inv(k_post * lambda_W)
            # mu_W = np.random.multivariate_normal(mu_post, covar_W)
            mu_W = np.random.multivariate_normal(mu0_W, covar_W)

            update('lambda_W')
            update('mu_W')

            # Update s across all users.
            cbias = c[iids]
            for i in uniq_uids:
                mask = train[uid] == i
                rated = mask.nonzero()[0]
                X_i = X[rated]
                y_i = y[rated]
                membs = P[uids]
                c_js = cbias[rated]

                alpha_si = alpha_s + X_i.shape[0] * alpha_G
                sigma_si = np.sqrt(1 / alpha_si)
                sum_js = np.sum(
                    y_i - c_js - np.sum(membs[rated].dot(W) * X_i, 1))
                mu_si = (alpha_s * mu_s + alpha_G * sum_js) / alpha_si
                s[i] = np.random.normal(mu_si, sigma_si)


            # Update c across all items.
            sbias = s[uids]
            membs = P[uids]
            for j in uniq_iids:
                mask = train[iid] == j
                rated = mask.nonzero()[0]
                X_j = X[rated]
                y_j = y[rated]
                s_is = sbias[rated]

                alpha_cj = alpha_c + X_i.shape[0] * alpha_G
                sigma_cj = np.sqrt(1 / alpha_cj)
                sum_is = np.sum(
                    y_j - s_is - np.sum(membs[rated].dot(W) * X_j, 1))
                mu_cj = (alpha_c * mu_c + alpha_G * sum_is) / alpha_cj
                c[j] = np.random.normal(mu_cj, sigma_c)


            # Update P for all users.
            cbias = c[iids]
            for i in uniq_uids:
                mask = train[uid] == i
                rated = mask.nonzero()[0]
                X_i = X[rated]
                y_i = y[rated]
                membs = P[uids]
                c_js = cbias[rated]

                lambda_Pi = lambda_P + alpha_G * W.dot(X_i.T.dot(X_i)).dot(W.T)
                covar_Pi = np.linalg.inv(lambda_Pi)
                sum_js = (X_i * (y_i - s[i] - c_js)[:, np.newaxis]).sum(axis=0)
                mu_Pi = covar_Pi.dot(lambda_P.dot(mu_P) + alpha_G * W.dot(sum_js))
                P[i] = np.random.multivariate_normal(mu_Pi, covar_Pi)


            # Update W
            sbias = s[uids]
            X_sum = X.T.dot(X)
            sum_term = (X * (cbias + sbias + y)[:, np.newaxis]).sum(axis=0)
            for l in xrange(L):
                P_l = P[:, l]
                lambda_Wl = lambda_W + alpha_G * (P_l ** 2).sum() * X_sum
                covar_Wl = np.linalg.inv(lambda_Wl)
                mu_Wl = covar_Wl.dot(lambda_W.dot(mu_W) + alpha_G * P_l.sum() * sum_term)
                W[l] = np.random.multivariate_normal(mu_Wl, covar_Wl)


        # Store samples
        for var in ['s', 'c', 'P', 'W']:
            update(var)

        # After thinning, make predictions for each sample.
        predictions[sample] = predict(alpha_G, s, c, P, W, X, uids, iids)
        rmse = np.sqrt(((predictions[sample] - y) ** 2).sum() / y.shape[0])
        logging.info('rmse: %.4f' % rmse)

        # Early stopping improvement threshold check.
        # if (prev_rmse - rmse) <= args.stopping_threshold:
        #     print '\tStopping threshold reached'
        #     break
        # else:
        #     prev_rmse = rmse

    y_pred = predict(alpha_G, s, c, P, W, test_X, test_uids, test_iids, (0, 4))
    rmse = np.sqrt(((y_pred - test_y) ** 2).sum() / test_y.shape[0])
    logging.info('Test RMSE: %.4f' % rmse)
