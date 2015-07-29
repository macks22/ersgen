"""
PMF Implementation in Python.

"""
import pandas as pd
import numpy as np

from util import predict, read_data


if __name__ == "__main__":
    np.random.seed(1234)

    epsilon = 50    # Learning rate
    lambda_  = 0.01  # Regularization parameter
    momentum = 0.8
    max_epoch = 50

    header_names, train, probe = read_data()
    uid, iid, target = header_names

    mean_rating = train[target].mean()
    ratings_test = probe[target].values.astype(np.double)

    num_batches = 9
    N = 100000  # number of training triplets processed per batch
    n = 6040    # number of users
    m = 3952    # number of items
    nf = 10     # number of features (latent factor vector dimensionality)

    # Randomly initialize user and item feature vectors.
    w_u = 0.1 * np.random.randn(n, nf)  # User
    w_i = 0.1 * np.random.randn(m, nf)  # Item

    # Allocate space for feature vector update vectors.
    w_u_update = np.zeros((n, nf))
    w_i_update = np.zeros((m, nf))

    err_train = np.zeros(max_epoch)
    err_valid = np.zeros(max_epoch)
    for epoch in range(max_epoch):
        rr = np.random.permutation(train.shape[0])  ## rr = random range
        train = train.ix[rr]

        for batch in range(num_batches):
            print 'epoch %d, batch %d' % (epoch + 1, batch + 1)

            train_subset = train.ix[range(batch*N, (batch+1)*N)]
            uids = train_subset[uid].values
            iids = train_subset[iid].values
            ratings = train_subset[target].values

            # Default prediction is the mean rating, so subtract it.
            ratings = ratings - mean_rating

            # Compute predictions.
            pred_out = np.sum(w_i[iids] * w_u[uids], 1)
            error = pred_out - ratings
            regular = np.sum(w_i[iids] ** 2 + w_u[uids] ** 2, 1)
            f = np.sum(error ** 2 + 0.5 * lambda_ * regular)

            # Compute gradients.
            IO = np.repeat(2 * error, nf).reshape(error.shape[0], nf)
            Ix_u = IO * w_i[iids] + lambda_ * w_u[uids]
            Ix_i = IO * w_u[uids] + lambda_ * w_i[iids]

            dw_u = np.zeros((n, nf))
            dw_i = np.zeros((m, nf))

            for ii in range(N):
                dw_u[uids[ii]] += Ix_u[ii]
                dw_i[iids[ii]] += Ix_i[ii]

            # Update user and item feature vectors.
            w_u_update = momentum * w_u_update + epsilon * (dw_u / N)
            w_u -= w_u_update

            w_i_update = momentum * w_i_update + epsilon * (dw_i / N)
            w_i -= w_i_update

        # Compute predictions after parameter updates.
        pred_out = np.sum(w_i[iids] * w_u[uids], 1)
        error = pred_out - ratings
        regular = np.sum(w_i[iids] ** 2 + w_u[uids] ** 2, 1)
        f_s = np.sum(error ** 2 + 0.5 * lambda_ * regular)
        err_train[epoch] = np.sqrt(f_s / N)

        # Compute predictions on the validation set.
        predictions = predict(w_u, w_i, mean_rating, probe)
        error = predictions - probe[target].values

        err_valid[epoch] = np.sqrt((error ** 2).sum() / probe.shape[0])
        print 'epoch %4d, batch %4d, Training RMSE: %6.4f, Test RMSE: %6.4f\n' % (
            epoch, batch, err_train[epoch], err_valid[epoch])

        if (epoch + 1) % 10 == 0:
            np.savetxt('w_u.csv', w_u, delimiter=',')
            np.savetxt('w_i.csv', w_i, delimiter=',')

    # Finally, save weights at the end.
    np.savetxt('w_u.csv', w_u, delimiter=',')
    np.savetxt('w_i.csv', w_i, delimiter=',')
