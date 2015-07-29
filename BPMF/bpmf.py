"""
Bayesian PMF (BPMF) Implementation in Python.

"""
import sys
import pandas as pd
import numpy as np

from util import predict, wishrnd, read_data, make_matrix


def cov(x):
    return np.cov(x, rowvar=0)


if __name__ == "__main__":
    np.random.seed(1234)

    max_epoch = 50
    iter_ = 0   # track iteration number
    n = 6040    # number of users
    m = 3952    # number of items
    nf = 10     # number of features (latent factor vector dimensionality)

    # Initialize hierarchical priors.
    beta = 2  # observation noise (precision)
    mu_u = np.zeros((nf, 1))
    mu_i = np.zeros((nf, 1))
    alpha_u = np.eye(nf)
    alpha_i = np.eye(nf)

    # Parameters of Inverse-Wishart distribution (User).
    W_u = np.eye(nf)
    b0_u = 2
    df_u = nf  # degrees of freedom
    mu0_u = np.zeros((nf, 1))

    # Parameters of Inverse-Wishart distribution (Item).
    W_i = np.eye(nf)
    b0_i = 2
    df_i = nf  # degrees of freedom
    mu0_i = np.zeros((nf, 1))

    header_names, train, probe = read_data()
    uid, iid, target = header_names

    mean_rating = train[target].mean()
    ratings_test = probe[target].values.astype(np.double)

    print 'Converting training triples to matrix format'
    args = header_names + [False]
    train_mat = make_matrix(train, *args)
    # train_mat = row_to_mat(train, *header_names)

    print 'Initializing BPMF using MAP solution from PMF'

    w_u_sample = np.loadtxt('w_u.csv', delimiter=',')
    w_i_sample = np.loadtxt('w_i.csv', delimiter=',')
    # err_test = np.empty((max_epoch, 1), dtype=object)

    # Initialization using MAP solution found by PMF.
    mu_u = w_u_sample.mean(axis=0)
    alpha_u = np.linalg.inv(cov(w_u_sample))

    mu_i = w_i_sample.mean(axis=0)
    alpha_i = np.linalg.inv(cov(w_i_sample))

    ratings_mat = train_mat.T
    probe_rat_all = predict(w_u_sample, w_i_sample, mean_rating, probe)
    counter_prob = 1
    print 'Done initializing'

    ngibbs = 2
    overall_err = np.zeros(max_epoch * ngibbs)
    prev_rmse = np.inf
    stopping_threshold = 0.00001

    for epoch in xrange(max_epoch):
        print '\nEpoch %d' % (epoch + 1)

        # Sample from movie hyperparameters.
        N = w_i_sample.shape[0]
        x_bar = np.mean(w_i_sample, 0).reshape(nf, 1)
        S_bar = cov(w_i_sample)

        tmp = mu0_i - x_bar
        W_post = np.linalg.inv(
            np.linalg.inv(W_i) + (N * S_bar) +
            N * b0_i * tmp.dot(tmp.T) / (b0_i + N))
        W_post = (W_post + W_post.T) / 2

        df_i_post = df_i + N
        alpha_i = wishrnd(W_post, df_i_post)
        mu_tmp = (b0_i * mu0_i + N * x_bar) / (b0_i + N)
        lam = np.linalg.cholesky(np.linalg.inv((b0_i + N) * alpha_i)).T
        mu_i = lam.dot(np.random.randn(nf)) + mu_tmp.reshape(nf)

        N = w_u_sample.shape[0]
        x_bar = np.mean(w_u_sample, 0).reshape(nf, 1)
        S_bar = cov(w_u_sample)

        # Sample from user hyperparameters.
        tmp = mu0_u - x_bar
        W_post = np.linalg.inv(
            np.linalg.inv(W_u) + (N * S_bar) +
            N * b0_u * tmp.dot(tmp.T) / (b0_u + N))
        W_post = (W_post + W_post.T) / 2

        df_u_post = df_u + N
        alpha_u = wishrnd(W_post, df_u_post)
        mu_tmp = (b0_u * mu0_u + N * x_bar) / (b0_u + N)
        lam = np.linalg.cholesky(np.linalg.inv((b0_u + N) * alpha_u)).T
        mu_u = lam.dot(np.random.randn(nf)) + mu_tmp.reshape(nf)

        # Start Gibbs updates over user/item feature vectors given hyperparams.
        for gibbs in range(ngibbs):
            print '\tGibbs sampling %d' % (gibbs + 1)

            # Infer posterior distribution over all item feature vectors.
            ratings_mat = ratings_mat.T
            for mm in range(m):
                # print 'item =%d' % mm
                # uids = (ratings_mat[:, mm] > 0).toarray().reshape(n)
                uids = ratings_mat[:, mm] > 0
                MM = w_u_sample[uids]
                # rr = (ratings_mat[uids, mm].toarray() -
                #       mean_rating).reshape(MM.shape[0])
                rr = ratings_mat[uids, mm] - mean_rating
                covar = np.linalg.inv(alpha_i + beta * MM.T.dot(MM))
                mean_i = covar.dot(beta * MM.T.dot(rr) + alpha_i.dot(mu_i))
                lam = np.linalg.cholesky(covar).T
                w_i_sample[mm] = lam.dot(np.random.randn(nf)) + mean_i

            # Infer posterior distribution over all user feature vectors.
            ratings_mat = ratings_mat.T
            for uu in range(n):
                # print 'user =%d' % uu
                # iids = (ratings_mat[:, uu] > 0).toarray().reshape(m)
                iids = ratings_mat[:, uu] > 0
                MM = w_i_sample[iids]
                # rr = (ratings_mat[iids, uu].toarray() -
                #       mean_rating).reshape(MM.shape[0])
                rr = ratings_mat[iids, uu] - mean_rating
                covar = np.linalg.inv(alpha_u + beta * MM.T.dot(MM))
                mean_u = covar.dot(beta * MM.T.dot(rr) + alpha_u.dot(mu_u))
                lam = np.linalg.cholesky(covar).T
                w_u_sample[uu] = lam.dot(np.random.randn(nf)) + mean_u

            probe_rat = predict(w_u_sample, w_i_sample, mean_rating, probe)
            probe_rat_all = ((counter_prob * probe_rat_all + probe_rat) /
                             (counter_prob + 1))
            counter_prob += 1

        # Make predictions on the validation data.
        error = ratings_test - probe_rat_all
        rmse = np.sqrt((error ** 2).sum() / probe.shape[0])

        iter_ += 1
        overall_err[iter_] = rmse
        print '\tAverage Test RMSE: %6.4f' % rmse

        if (prev_rmse - rmse) <= stopping_threshold:
            print '\tStopping threshold reached'
            break
        else:
            prev_rmse = rmse
