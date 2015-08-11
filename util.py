import os
import logging

try:
    import ujson as json
except ImportError:
    import json

import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse


def blas_cov(X):
    n,p = X.shape
    m = X.mean(axis=0)
    # covariance matrix with correction for rounding error
    # S = (cx'*cx - (scx'*scx/n))/(n-1)
    # Am Stat 1983, vol 37: 242-247.
    cx = X - m
    scx = cx.sum(axis=0)
    scx_op = dger(-1.0/n,scx,scx)
    S = dgemm(1.0, cx.T, cx.T, beta=1.0,
            c=scx_op, trans_a=0, trans_b=1, overwrite_c=1)
    S[:] *= 1.0/(n-1)
    return S.T

# function [pred_out] = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
# 
# %%% Make predicitions on the validation data
# 
#  aa_p   = double(probe_vec(:,1));
#  aa_m   = double(probe_vec(:,2));
#  rating = double(probe_vec(:,3));
# 
#  pred_out = sum(w1_M1_sample(aa_m,:).*w1_P1_sample(aa_p,:),2) + mean_rating;
#  ff = find(pred_out>5); pred_out(ff)=5;
#  ff = find(pred_out<1); pred_out(ff)=1;

def predict(w_u, w_i, mean_rating, data, uid='uid', iid='iid', target='target',
            bounds=(0, 5)):
    """Make predictions given user/item weights and the mean rating."""
    uids = data[uid]
    iids = data[iid]
    ratings = data[target]

    lo, hi = bounds
    predictions = np.sum(w_u[uids] * w_i[iids], 1) + mean_rating
    predictions[predictions < lo] = lo
    predictions[predictions > hi] = hi
    return predictions


def wishrnd(W, df):
    """Generate wishart random variables.

    Args:
        W: np.ndarray
            the scale matrix
        df: int
            degrees of freedom
    """
    W = np.array(W)
    n = W.shape[0]

    # obtain lower triangle of matrix
    try:
        L = np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        p, L, u = sp.linalg.lu(W)

    # borrowed from Matlab
    if (df <= 81 + n) and (df == round(df)):
        # direct
        X = np.dot(L, np.random.normal(size=(n, df)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(df - np.arange(0, n), size=n)))
        A[np.tri(n, k=-1, dtype=bool)] = \
            np.random.normal(size=(n * (n - 1) / 2.))
        X = np.dot(L, A)

    return np.dot(X, X.T)


def read_data(uid='uid', iid='iid', target='target'):
    """Read the test data files (train, probe), adjust indices, and return."""
    header_names = [uid, iid, target]
    train = pd.read_csv('train_data.csv', header=None, names=header_names)
    probe = pd.read_csv('probe_data.csv', header=None, names=header_names)

    # Adjust ids for 0-indexing.
    for dataset in [train, probe]:
        for key in [uid, iid]:
            dataset.loc[:, key] = dataset[key] - 1

    return header_names, train, probe


def row_to_mat(data, uid='uid', iid='iid', target='target'):
    """Convert the data from row to matrix format."""
    mat = data.pivot(uid, iid, target)\
              .reindex(np.arange(data[uid].max()+1))

    for col in np.arange(data[iid].max()+1):
        if not col in mat.columns:
            mat[col] = 0

    return mat.sort_index(axis=1)\
              .fillna(0)

# %% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  
# 
# load moviedata
# 
# num_m = 3952;
# num_p = 6040;
# count = zeros(num_p,num_m,'single'); %for Netflida data, use sparse matrix instead. 
# 
# for mm=1:num_m
#  ff= find(train_vec(:,2)==mm);
#  count(train_vec(ff,1),mm) = train_vec(ff,3);
# end 

def make_matrix(data, uid='uid', iid='iid', target='target', sparse=False):
    """Convert the data from row to matrix format.
    This is a direct translation from the MATLAB BPMF code."""
    n = data[uid].max()+1
    m = data[iid].max()+1

    if sparse:
        count = scipy.sparse.lil_matrix((n, m), dtype=np.int)
        index = lambda idx: data.loc[idx, target].reshape(idx.sum(), 1)
    else:
        count = np.zeros((n, m), dtype=np.int)
        index = lambda idx: data.loc[idx, target]

    for itemid in xrange(m):
        idx = data[iid] == itemid
        count[data.loc[idx, uid], itemid] = index(idx)

    if sparse:
        return scipy.sparse.csr_matrix(count).astype(np.float)
    else:
        return count.astype(np.float)


def scale_features(data, features):
    """Scale all `features` using Z-score scaling."""
    logging.info('performing z-score scaling on features')
    logging.debug('%s' % ', '.join(features))
    for f in features:
        data[f] = (data[f] - data[f].mean()) / data[f].std(ddof=0)


def add_squared_features(data, features):
    """Add squared versions of the given features."""
    logging.info('adding %d quadratic features' % len(features))
    new_keys = []
    for f in features:
        key = '%s^2' % f
        data[key] = data[f] ** 2
        new_keys.append(key)
    return new_keys


def map_ids(data, key):
    """Map ids to 0-contiguous index. This enables the use of these ids as
    indices into an array (for the bias terms, for instance). This returns the
    number of unique IDs for `key`.
    """
    ids = data[key].unique()
    n = len(ids)
    id_map = dict(zip(ids, range(n)))
    data[key] = data[key].apply(lambda _id: id_map[_id])
    return n




def save_np_vars(vars, savedir):
    """Save a dictionary of numpy variables to `savedir`. We assume
    the directory does not exist; an OSError will be raised if it does.
    """
    logging.info('writing numpy vars to directory: %s' % savedir)
    os.mkdir(savedir)
    shapes = {}
    for varname in vars:
        data = vars[varname]
        var_file = os.path.join(savedir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape

        ## Store shape information for reloading.
        shape_file = os.path.join(savedir, 'shapes.json')
        with open(shape_file, 'w') as sfh:
            json.dump(shapes, sfh)


def load_np_vars(savedir):
    """Load numpy variables saved with `save_np_vars`."""
    shape_file = os.path.join(savedir, 'shapes.json')
    with open(shape_file, 'r') as sfh:
        shapes = json.load(sfh)

    vars = {}
    for varname, shape in shapes.items():
        var_file = os.path.join(savedir, varname + '.txt')
        vars[varname] = np.loadtxt(var_file).reshape(shape)

    return vars
