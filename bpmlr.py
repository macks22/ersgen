"""
An implementation of the Bayesian formulation of the Personalized Multi-Linear
Regression (PMLR) model: Bayesian PMLR (BPMRL).

"""
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn import preprocessing

from util import wishrnd


# Error codes
BOUNDS_FORMAT = 1000
MISSING_ATTRIBUTE = 1001


def predict(s, c, P, W, X, uids, iids, bounds=(0, 4)):
    """Make predictions using BPMLR given parameters.
    Assumes features have already been scaled appropriately.
    """
    lo, hi = bounds
    sbias = s[uids]
    cbias = c[iids]
    membs = P[uids]

    predictions = sbias + cbias + np.sum(membs.dot(W) * X, 1)
    predictions[predictions < lo] = lo
    predictions[predictions > hi] = hi
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

    # TODO: Init params from MLR model results.

    # Init s params and hyperparams.
    mu0_s = 0
    alpha0_s = 1
    beta0_s = 1
    k0_s = 1

    alpha_s = np.random.gamma(alpha0_s, beta0_s)
    sigma_s = np.sqrt(1 / (k0_s * alpha_s))
    mu_s = np.random.normal(mu0_s, sigma_s)

    # Init c params and hyperparams.
    mu0_c = 0
    alpha0_c = 1
    beta0_c = 1
    k0_c = 1

    alpha_c = np.random.gamma(alpha0_c, beta0_c)
    sigma_c = np.sqrt(1 / (k0_c * alpha_c))
    mu_c = np.random.normal(mu0_c, sigma_c)

    # Init P params and hyparparams.
    mu0_P = np.zeros(L)
    df0_P = L
    W0_P = np.eye(L)
    k0_P = 1

    lambda_P = wishrnd(W0_P, df0_P)
    covar_P = np.linalg.inv(lambda_P * k0_P)
    mu_P = np.random.multivariate_normal(mu0_P, covar_P)

    # Init W params and hyperparams.
    mu0_W = np.zeros(nf)
    df0_W = nf
    W0_W = np.eye(nf)
    k0_W = 1

    lambda_W = wishrnd(W0_W, df0_W)
    covar_W = np.linalg.inv(lambda_W * k0_W)
    mu_W = np.random.multivariate_normal(mu0_W, covar_W)

    # Finally, init actual parameters.
    # TODO: init from least squares MLR solution.
    s = np.random.normal(mu_s, np.sqrt(1 / alpha_s), N)
    c = np.random.normal(mu_c, np.sqrt(1 / alpha_c), M)
    P = np.random.multivariate_normal(mu_P, covar_P, N)
    W = np.random.multivariate_normal(mu_W, covar_W, L)

    # Calculate initial predictions.
    overall_err = np.zeros(args.nsamples)
    nsamples = args.nsamples - args.burnin
    predictions = np.ndarray((nsamples, nd_test))

    initial = predict(s, c, P, W, test_X, test_uids, test_iids, (0, 4))
    prev_rmse = np.sqrt(((initial - test_y) ** 2).sum() / test_y.shape[0])


    # Begin Gibbs sampling.
    for sample in xrange(args.nsamples):
        print 'sample %d' % sample
        for sub_sample in xrange(args.thin):

            # Update s hyperparams...
            s_mean = s.mean()
            beta0_s += (0.5 * (
                        ((s - s_mean) ** 2).sum() +
                        (N * k0_s) * ((s_mean - mu0_s) ** 2) / (k0_s + N)))
            mu0_s = (k0_s * mu0_s + N * s_mean) / (k0_s + N)
            # k0_s += N
            alpha0_s += N / 2

            # ...and params.
            alpha_s = np.random.gamma(alpha0_s, beta0_s)
            sigma_s = np.sqrt(1 / (k0_s * alpha_s))
            mu_s = np.random.normal(mu0_s, sigma_s)

            # Update c hyperparams...
            c_mean = c.mean()
            beta0_c += (0.5 * (
                        ((c - c_mean) ** 2).sum() +
                        (M * k0_c) * ((c_mean - mu0_c) ** 2) / (k0_c + M)))
            mu0_c = (k0_c * mu0_c + M * c_mean) / (k0_c + M)
            # k0_c += M
            alpha0_c += M / 2

            # ...and params.
            alpha_c = np.random.gamma(alpha0_c, beta0_c)
            sigma_c = np.sqrt(1 / (k0_c * alpha_c))
            mu_c = np.random.normal(mu0_c, sigma_c)

            # Update P hyperparams...
            P_hat = P.mean(axis=0)
            dev_P = P - P_hat
            mu_tmp = (P_hat - mu0_P).reshape(P_hat.shape[0], 1)
            W0_P = np.linalg.inv(
                np.linalg.inv(W0_P) + dev_P.T.dot(dev_P) +
                (k0_P * N) * mu_tmp.dot(mu_tmp.T) / (k0_P + N))

            mu0_P = (k0_P * mu0_P + N * P_hat) / (k0_P + N)
            df0_P += N
            # k0_P += N

            # ...and params.
            lambda_P = wishrnd(W0_P, df0_P)
            covar_P = np.linalg.inv(lambda_P * k0_P)
            mu_P = np.random.multivariate_normal(mu0_P, covar_P)

            # Update W hyperparams...
            W_hat = W.mean(axis=0)
            dev_W = W - W_hat
            mu_tmp = (W_hat - mu0_W).reshape(W_hat.shape[0], 1)
            W0_W = np.linalg.inv(
                np.linalg.inv(W0_W) + dev_W.T.dot(dev_W) +
                (k0_W * L) * mu_tmp.dot(mu_tmp.T) / (k0_W + L))
            W0_W = (W0_W + W0_W.T) / 2  # Not sure what this is about.

            mu0_W = (k0_W * mu0_W + L * W_hat) / (k0_W + L)
            df0_W += L
            # k0_W += L

            # ...and params.
            lambda_W = wishrnd(W0_W, df0_W)
            covar_W = np.linalg.inv(lambda_W * k0_W)
            mu_W = np.random.multivariate_normal(mu0_W, covar_W)


            # Update parameters across users: s and P
            cbias = c[iids]
            for i in uniq_uids:
                mask = train[uid] == i
                rated = mask.nonzero()[0]
                X_i = X[rated]
                y_i = y[rated]
                membs = P[uids]
                c_js = cbias[rated]

                # Update s
                alpha_si = alpha_s + X_i.shape[0] * alpha_G
                sigma_si = np.sqrt(1 / alpha_si)
                sum_js = np.sum(
                    c_js + np.sum(membs[rated].dot(W) * X_i, 1) - y_i)
                mu_si = (alpha_s * mu_s - alpha_G * sum_js) / alpha_si
                s[i] = np.random.normal(mu_si, sigma_si)

                # Update P
                lambda_Pi = lambda_P - alpha_G * (X_i ** 2).sum()
                covar_Pi = np.linalg.inv(lambda_Pi)
                sum_js = (X_i * np.repeat(y_i + s[i] + c_js, nf)\
                                  .reshape(X_i.shape[0], nf)).sum(axis=0)
                mu_Pi = covar_Pi.dot(lambda_P.dot(mu_P) - alpha_G * W.dot(sum_js))
                P[i] = np.random.multivariate_normal(mu_Pi, covar_Pi)


            # Update parameters across items: c
            sbias = s[uids]
            membs = P[uids]
            for j in uniq_iids:
                mask = train[iid] == j
                rated = mask.nonzero()[0]
                X_j = X[rated]
                y_j = y[rated]
                s_is = sbias[rated]

                # Update c
                alpha_cj = alpha_c + X_i.shape[0] * alpha_G
                sigma_cj = np.sqrt(1 / alpha_cj)
                sum_is = np.sum(
                    s_is + np.sum(membs[rated].dot(W) * X_j, 1) - y_j)
                mu_cj = (alpha_c * mu_c - alpha_G * sum_is) / alpha_cj
                c[j] = np.random.normal(mu_cj, sigma_cj)


            # Update W
            P_star = P.sum(axis=0)
            X_sq_sum = (X ** 2).sum()
            sum_term = np.sum(
                X * (cbias + sbias + y).repeat(nf).reshape(X.shape[0], nf), 0)
            for l in xrange(L):
                tmp = P_star[l] * alpha_G
                lambda_Wl = lambda_W - tmp * P_star[l] * X_sq_sum
                covar_Wl = np.linalg.inv(lambda_Wl)
                mu_Wl = covar_Wl.dot(lambda_W.dot(mu_W) - tmp * sum_term)
                W[l] = np.random.multivariate_normal(mu_Wl, covar_Wl)


        # After thinning, make predictions for each sample.
        predictions[sample] = predict(
            s, c, P, W, test_X, test_uids, test_iids, (0, 4))
        rmse = np.sqrt(((predictions - test_y) ** 2).sum() / test_y.shape[0])

        # Early stopping improvement threshold check.
        if (prev_rmse - rmse) <= args.stopping_threshold:
            print '\tStopping threshold reached'
            break
        else:
            prev_rmse = rmse
