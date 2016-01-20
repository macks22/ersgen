"""
Mixed-membership Multi-Linear Regression model (non-Theano version).

"""
import logging
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

from cmlr import (
    fit_mlr_sgd, fit_mlr_sgd_nn, fit_mlr_als, rmse_from_err, compute_rmse)
from util import (
    scale_features, add_squared_features, map_ids, save_np_vars, load_np_vars)


def split_train_test(data, term):
    return data[data.term < term], data[data.term == term]


def make_parser():
    parser = argparse.ArgumentParser(
        description="mixed-membership multi-linear regression")
    parser.add_argument(
        '-m', '--method',
        choices=('sgd', 'als'), default='sgd',
        help='learning method to use')
    parser.add_argument(
        '-d', '--data_file', default='',
        help='path of data file')
    parser.add_argument(
        '-l', '--nmodels',
        type=int, default=3,
        help='number of linear regression models')
    parser.add_argument(
        '-lw', '--lambda-w',
        type=float, default=0.01,
        help='regularization multiplier for P and W')
    parser.add_argument(
        '-lb', '--lambda-b',
        type=float, default=0.001,
        help='regularization multiplier for s and c')
    # parser.add_argument(
    #     '-r', '--regularization',
    #     choices=('fro', 'l1', 'l2'), default='fro',
    #     help='type of regularization to use')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-i', '--iters',
        type=int, default=10,
        help='number of iterations')
    parser.add_argument(
        '-s', '--std',
        type=float, default=0.01,
        help='standard deviation of Gaussian noise used in model param'
             'initialization')
    parser.add_argument(
        '-q', '--quadratic',
        action='store_true', default=False,
        help='add quadratic features (power 2)')
    parser.add_argument(
        '-t', '--target',
        default='grade',
        help='target variable to predict; default is "grade"')
    parser.add_argument(
        '-n', '--nonneg',
        action='store_true', default=False,
        help='enable non-negativity constraints on all params')
    parser.add_argument(
        '-v', '--verbose',
        type=int, default=0, choices=(0, 1, 2),
        help='verbosity level; 0=None, 1=INFO, 2=DEBUG')
    parser.add_argument(
        '-o', '--output',
        default='',
        help='directory to save model params to; default is none: do not save')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose == 2 else
               logging.INFO if args.verbose == 1 else
               logging.ERROR),
        format="[%(asctime)s]: %(message)s")

    uid = 'sid'
    iid = 'cid'
    # features = ['term', 'gender', 'age', 'schrs', 'hsgpa', 'cum_gpa',
    #             'cum_cgpa', 'chrs', 'term_chrs', 'term_enrolled']
    data_keys = [uid, iid, args.target]
    # to_read = list(set(features + data_keys))

    logging.info('reading train/test data')
    # logging.info('reading columns: %s' % ', '.join(to_read))
    # data = pd.read_csv(args.data_file, usecols=to_read)
    data = pd.read_csv(args.data_file)
    features = list(set(data.columns) - set(data_keys))
    logging.info('read columns: %s' % ', '.join(data.columns))

    # Add quadratic features of power 2.
    if args.quadratic:
        features += add_squared_features(data, features)

    # Map user/item ids to bias indices.
    N = map_ids(data, uid)
    M = map_ids(data, iid)

    # Split dataset into train & test, predicting only for last term (for now).
    train, test = split_train_test(data, data.term.max())

    def split_data(df):
        """Split data into X, y, uids, iids."""
        return (df.drop(data_keys, axis=1).values,
                df[args.target].values,
                df[uid].values,
                df[iid].values)

    logging.info('splitting train/test sets into X, y')
    train_x, train_y, train_uids, train_iids = split_data(train)
    test_x, test_y, test_uids, test_iids = split_data(test)

    # Z-score scaling to mean of 0 and variance of 1.
    # scale_features(data, features)
    scaler = preprocessing.StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    # Get dimensions of training data.
    nd, nf = train_x.shape
    l = args.nmodels

    logging.info('%d users, %d items' % (N, M))
    logging.info('%d dyads with %d features' % (nd, nf))
    logging.info('l=%d, lr=%f' % (l, args.lrate))

    # Multiplex learning method.
    params = dict(_X=train_x,
                  y=train_y,
                  uids=train_uids,
                  iids=train_iids,
                  l=l,
                  lambda_w=args.lambda_w,
                  lambda_b=args.lambda_b,
                  iters=args.iters,
                  std=args.std,
                  verbose=args.verbose)

    if args.method == 'als':
        if args.nonneg:
            print 'Non-negative ALS learning not implemented'
            sys.exit(1)
        else:
            fit = fit_mlr_als
    else: # args.method == 'sgd'
        fit = fit_mlr_sgd_nn if args.nonneg else fit_mlr_sgd
        params['lrate'] = args.lrate

    # Train the model using the selected learning method.
    model = fit(**params)

    logging.info('making predictions')
    print 'MLR RMSE:\t%.4f' % compute_rmse(
        model, test_x, test_y, test_uids, test_iids)

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(train_y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)

    # Save model params.
    if args.output:
        save_np_vars(model, args.output)


    # Calculate feature importance metrics.
    # nd = number of nonzero observations
    uniq_uids = np.unique(train_uids)
    item_count = np.vectorize(lambda i: train[train[uid] == i].shape[0])
    m = item_count(uniq_uids)

    user_count = np.vectorize(lambda j: train[train[iid] == j].shape[0])
    uniq_iids = np.unique(train_iids)
    n = user_count(uniq_iids)

    # Extract model params from returned dict.
    s = model['s']
    c = model['c']
    P = model['P']
    W = model['W']

    # Calculate individual deviation contributions.
    dev = {}

    # user and item bias terms are simple.
    dev['s'] = (m * s).sum()
    dev['c'] = (n * c).sum()

    # For the regression coefficients, we actually need to sum over i and j.
    dev['W'] = np.zeros(nf)
    membs = P[train_uids]
    sbias = s[train_uids]
    cbias = c[train_iids]
    for i in xrange(train_x.shape[0]):
        dev['W'] += abs(membs[i].T.dot(W)[0] * train_x[i])

    # Calculate total absolute deviation over all records.
    T = dev['s'] + dev['c'] + dev['W'].sum()

    # Now calculate importances.
    imp = {k: dev[k] / T for k in dev}

    colname = 'Importance'
    I = pd.DataFrame(imp['W'], index=train.drop(data_keys, axis=1).columns)\
          .rename(columns={0: colname})
    I.ix[uid] = imp['s']
    I.ix[iid] = imp['c']
    I = I.sort(colname, ascending=False)

    # Plot feature importance.
    deep_blue = sns.color_palette('colorblind')[0]
    ax = sns.barplot(data=I, x=colname, y=I.index, color=deep_blue)
    ax.set(title='Feature Importance for Grade Prediction',
           ylabel='Feature',
           xlabel='Proportion of Deviation From Intercept')
    ax.figure.show()
