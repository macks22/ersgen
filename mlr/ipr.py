import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn import preprocessing

from util import save_model_vars, load_model_vars
from cipr import (
    fit_ipr_sgd, compute_errors, compute_rmse, ipr_predict, rmse_from_err)


# Error codes
BOUNDS_FORMAT = 1000
MISSING_ATTRIBUTE = 1001
DIM_MISMATCH = 1002
BAD_FEATURE_CONF = 1003
BAD_FILENAME = 1004
BAD_CMDLINE_ARG = 1005


class BadFeatureConfig(Exception):
    """Raise when bad feature configuration file is found."""
    pass

class UnlearnedModel(Exception):
    """Raise when trying to predict with unlearned model."""

    def __init__(self, message, params):
        """params is a list of params not yet learned."""
        self.message = message
        self.params = params

    def __repr__(self):
        return "UnlearnedModel: %s [%s]" % (
            self.message, ', '.join(self.params))

    def __str__(self):
        return self.__repr__()


def read_feature_guide(fname):
    """Read the feature guide and parse out the specification.

    The expected file format is the following:

        t:<target>;
        e:<comma-separated categorical entity names>;
        c:<comma-separated categorical variable names>;
        r:<comma-separated real-valued variable names>;

    Whitespace is ignored, as are lines that start with a "#" symbol. Any
    variables not included in one of the three groups is ignored. We assume the
    first two categorical variables are the user and item ids.

    Args:
        fname (str): Path of the file containing the feature guide.

    Returns:
        iterable: (target, entity list, categoricals list, real-valued list).

    """
    with open(fname) as f:
        lines = [l.strip() for l in f.read().split('\n')
                 if not l.startswith('#') and l.strip()]

    parsing = False
    keys = ['t', 'e', 'c', 'r']
    vars = {var: [] for var in keys}
    for line in lines:
        if not parsing:
            k, csv = line.split(':')
        else:
            csv = line

        vars[k].extend([val.strip() for val in csv.split(',')])
        parsing = not line.endswith(';')
        if not parsing:
            vars[k][-1] = vars[k][-1][:-1]

    # Remove whitespace strings. These may have come from something like:
    # c: this, , that;
    for k in keys:
        vars[k] = [val for val in vars[k] if val]  # already stripped

    # Sanity checks.
    num_targets = len(vars['t'])
    if num_targets != 1:
        raise BadFeatureConfig(
            'feature conf should specify 1 target; got %d; check for'
            'the semi-colon at the end of the t:<target> line' % num_targets)

    num_bias = len(vars['e'])
    if not num_bias:
        raise BadFeatureConfig('no entity variables given; need at least 1')

    num_features = len(vars['c']) + len(vars['r'])
    if not num_features > 0:
        raise BadFeatureConfig('no predictors specified')

    spec = (vars['t'][0], vars['e'], vars['c'], vars['r'])
    target, ents, cats, reals = spec
    logging.info('read the following feature guide:')
    logging.info('target: %s' % target)
    logging.info('entities: %s' % ', '.join(ents))
    logging.info('categoricals: %s' % ', '.join(cats))
    logging.info('real-valueds: %s' % ', '.join(reals))
    return spec


def map_ids(data, key, id_map=None):
    """Map ids to 0-contiguous index. This enables the use of these ids as
    indices into an array (for the bias terms, for instance). This returns the
    number of unique IDs for `key`.
    """
    if id_map is None:
        ids = data[key].unique()
        n = len(ids)
        id_map = dict(zip(ids, range(n)))

    data[key] = data[key].apply(lambda _id: id_map[_id])
    return id_map


def read_train_test(train_file, test_file, conf_file):
    """Read the train and test data according to the feature guide in the
    configuration file. Return the train and test data as (X, y) pairs for the
    train and test data.

    Args:
        train_file (str): Path of the CSV train data file.
        test_file (str): Path of the CSV test data file.
        conf_file (str): Path of the configuration file.

    Returns:
        tuple: (train_eids, train_X, train_y,
                test_eids, test_X, test_y,
                feat_indices, number_of_entities),
               preprocessed train/test data and mappings from features to
               indices in the resulting data matrices. Both train and test are
               accompanied by 0-contiguous primary entity id arrays.

    """
    try:
        target, ents, cats, reals = read_feature_guide(conf_file)
    except IOError:
        raise IOError('invalid feature guide conf file path: %s' % conf_file)

    to_read = [target] + ents + cats + reals
    def read_file(name, fpath):
        try:
            data = pd.read_csv(fpath, usecols=to_read)
        except IOError:
            raise IOError('invalid %s file path: %s' % (name, fpath))
        except ValueError as err:
            attr_name = err.args[0].split("'")[1]
            attr_type = ('entity' if attr_name in ents else
                         'categorical' if attr_name in cats else
                         'real-valued' if attr_name in reals else
                         'target')
            raise BadFeatureConfig('%s attribute %s not found in %s file' % (
                attr_type, attr_name, name))

        return data

    train = read_file('train', train_file)
    test = read_file('test', test_file)
    nd_train = train.shape[0]
    nd_test = test.shape[0]
    logging.info('number of instances: train=%d, test=%d' % (
        train.shape[0], test.shape[0]))

    return train, test, target, ents, cats, reals


def preprocess(train, test, target, ents, cats, reals):
    """Return preprocessed (X, y, eid) pairs for the train and test sets.

    Preprocessing includes:

    1.  Map primary entity ID (first in ents) to a 0-contiguous range.
    2.  Z-score scale the real-valued features.
    3.  One-hot encode the categorical features (including primary entity ID).

    This function tries to be as general as possible to accomodate learning by
    many models. As such, there are 8 return values. The first three are:

    1.  train_eids: primary entity IDs as a numpy array
    2.  train_X: training X values (first categorical, then real-valued)
    3.  train_y: training y values (unchanged from input)

    The next three values are the same except for the test set. The final two
    values are:

    7.  indices: The indices of each feature in the encoded X matrix.
    8.  nents: The number of categorical features after one-hot encoding.

    """

    # Separate X, y for train/test data.
    nd_train = train.shape[0]
    nd_test = test.shape[0]
    train_y = train[target].values
    test_y = test[target].values

    # Read out id lists for primary entity.
    id_map = map_ids(train, ents[0])
    map_ids(test, ents[0], id_map)
    train_eids = train[ents[0]].values
    test_eids = test[ents[0]].values

    # Z-score scaling of real-valued features.
    scaler = preprocessing.StandardScaler()
    train_reals = scaler.fit_transform(train[reals])
    test_reals = scaler.transform(test[reals])

    # One-hot encoding of entity and categorical features.
    catf = ents + cats
    all_cats = pd.concat((train[catf], test[catf]))
    encoder = preprocessing.OneHotEncoder()
    enc = encoder.fit_transform(all_cats)
    train_cats = enc[:nd_train]
    test_cats = enc[nd_train:]

    # Create a feature map for decoding one-hot encoding.
    ncats = encoder.active_features_.shape[0]
    nreal = train_reals.shape[1]
    nf = ncats + nreal

    # Count entities.
    logging.info('after one-hot encoding, found # unique values:')
    counts = np.array([
        all_cats[catf[i]].unique().shape[0]
        for i in xrange(len(catf))
    ])
    indices = zip(catf, np.cumsum(counts))
    for attr, n_values in zip(ents, counts):
        logging.info('%s: %d' % (attr, n_values))

    # Add in real-valued feature indices.
    indices += zip(reals, range(indices[-1][1] + 1, nf + 1))

    # How many entity features and categorical features do we have?
    nents = dict(indices)[ents[-1]]
    ncats = ncats - nents
    nf = nents + ncats + nreal

    ent_idx = range(len(ents))
    cat_idx = range(len(ents), len(ents) + len(cats))
    nactive_ents = sum(encoder.n_values_[i] for i in ent_idx)
    nactive_cats = sum(encoder.n_values_[i] for i in cat_idx)

    logging.info('number of active entity features: %d of %d' % (
        nents, nactive_ents))
    logging.info('number of active categorical features: %d of %d' % (
        ncats, nactive_cats))
    logging.info('number of real-valued features: %d' % nreal)

    # Put all features together.
    train_X = sp.sparse.hstack((train_cats, train_reals))
    test_X = sp.sparse.hstack((test_cats, test_reals))
    logging.info('Total of %d features after encoding' % nf)

    return (train_eids, train_X, train_y,
            test_eids, test_X, test_y,
            indices, nents)


class Model(object):
    """General model class with load/save/preprocess functionality."""

    def set_fguide(self, fguidef=''):
        if fguidef:
            self.read_fguide(fguidef)
        else:
            self.reset_fguide()

    def read_fguide(self, fguidef):
        self.target, self.ents, self.cats, self.reals = \
            read_feature_guide(fguidef)

    def reset_fguide(self):
        self.target = ''
        for name in ['ents', 'cats', 'reals']:
            setattr(self, name, [])

    @property
    def fgroups(self):
        return ('ents', 'cats', 'reals')

    @property
    def nf(self):
        return sum(len(group) for group in self.fgroups)

    def check_fguide(self):
        if self.nf <= 0:
            raise ValueError("preprocessing requires feature guide")

    def read_necessary_fguide(self, fguide=''):
        if fguide:
            self.read_fguide(fguide)
        else:
            check_fguide()

    def preprocess(self, train, test, fguidef=''):
        self.read_necessary_fguide(fguidef)
        return preprocess(
            train, test, self.target, self.ents, self.cats, self.reals)

    def check_if_learned(self):
        unlearned = [attr for attr, val in self.model.items() if val is None]
        if len(unlearned) > 0 or len(self.model) == 0:
            raise UnlearnedModel("IPR predict with unlearned params", unlearned)

    def save(self, savedir, ow=False):
        save_model_vars(self.model, savedir, ow)

    def load(self, savedir):
        self.model = load_model_vars(savedir)



class IPR(Model):
    """Individualized Profile Regression model."""

    def __init__(self, nmodels, lambda_w=0.01, lambda_b=0.0, iters=10,
                 lrate=0.001, epsilon=0.00001, init_std=0.01, nonneg=0,
                 verbose=0, fguidef=''):
        """Initialize the model. This sets all parameters that govern learning.
        If the files are passed in, they are read and the data is cached in the
        initialized object.

        See the make_parser method for detail on parameters.
        """
        self.nmodels = nmodels
        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.iters = iters
        self.lrate = lrate
        self.epsilon = epsilon
        self.init_std = init_std
        self.nonneg = nonneg
        self.verbose = verbose

        # Set up feature guide
        self.set_fguide(fguidef)

        # all model params initially set to None
        self.model = {attr: None for attr in self.param_names}

    @property
    def args_suffix(self):
        parts = [
            'k%d' % self.nmodels,
            'lw%.4f' % self.lambda_w,
            'lb%.4f' % self.lambda_b,
            'i%d' % self.iters,
            'lr%.4f' % self.lrate,
            's%.4f' % self.init_std]
        if self.nonneg:
            parts.append('nn')

        return '-'.join(parts)

    @property
    def param_names(self):
        return ('w0', 'w', 'P', 'W')

    @property
    def w0(self):
        return self.model['w0']

    @property
    def w(self):
        return self.model['w']

    @property
    def P(self):
        return self.model['P']

    @property
    def W(self):
        return self.model['W']

    def fit(self, X, y, eids, nb):
        self.model = fit_ipr_sgd(
            X, y, eids, nb,
            k=self.nmodels,
            lambda_w=self.lambda_w,
            lambda_b=self.lambda_b,
            iters=self.iters,
            std=self.init_std,
            nn=self.nonneg,
            verbose=self.verbose,
            lrate=self.lrate,
            eps=self.epsilon)

    def predict(self, X, eids, nb):
        self.check_if_learned()
        logging.info('making IPR predictions')
        return ipr_predict(self.model, X.tocsc(), eids, nb)

    def feature_importance(self, X, y, eids, nb, train, f_indices, fguidef=''):
        """Calculate feature importance metrics."""
        self.check_if_learned()
        self.read_necessary_fguide(fguidef)

        X = X.tocsr()
        X_ = X[:, nb:]  # n x p

        # Extract model params from dict. No need for w0.
        w = self.model['w']
        P = self.model['P']
        W = self.model['W']

        # Read in training data for use of DataFrame.
        nents = len(self.ents)  # number of entities

        n, nf = X.shape  # num training examples and num features
        p = nf - nb  # num non-entity predictor variables
        k = self.nmodels
        idx_table = dict(f_indices)

        # Deviation calculations will be stored here.
        dev = {}

        # Start by taking absolute values of all parameters.
        w = abs(w)
        P = abs(P)
        W = abs(W)
        X_ = abs(X_)

        # Calculate deviations for the entity bias terms.
        bi_prev = 0
        for i, ent in enumerate(self.ents):
            b_i = idx_table[ent]
            weights = w[bi_prev: b_i]
            bi_prev = b_i

            uniq_ids = np.unique(train[ent])
            nz_count = np.vectorize(lambda i: train[train[ent] == 1].shape[0])
            nnz = nz_count(uniq_ids)
            dev[ent] = (weights * nnz).sum()

        # Next calculate deviation for the rest of the features.
        membs = P[eids]  # n x k
        fdev_pprof = X_.T.dot(membs).T * W  # dev per profile
        fdev = fdev_pprof.sum(axis=0)  # dev across profiles

        n_prev = 0
        for f, fname in enumerate(self.cats + self.reals):
            n_f = idx_table[fname] - nb
            dev[fname] = fdev[n_prev: n_f].sum()
            n_prev = n_f

        # Calculate total absolute deviation over all records.
        T = sum(dev.values())

        # Now calculate importances.
        colname = 'Importance'
        imp = {k: dev[k] / T for k in dev}
        I = pd.DataFrame(imp.values(), index=imp.keys(), columns=[colname])
        I = I.sort(colname, ascending=False)

        # Calculate profile contributions.
        membs = P[eids]         # n x k
        reg = X_.dot(W.T)       # n x k
        contribs = membs * reg  # n x k
        contribs /= contribs.sum(axis=1)[:, np.newaxis]

        # Calculate feature importance per profile.
        devs = [{} for _ in xrange(k)]
        n_prev = 0
        for f, fname in enumerate(self.cats + self.reals):
            n_f = idx_table[fname] - nb
            for l in xrange(k):
                devs[l][fname] = fdev_pprof[l][n_prev: n_f].sum()
            n_prev = n_f

        # Add in entity bias deviations.
        for l in xrange(k):
            for ent in self.ents:
                devs[l][ent] = dev[ent]

        imp = [{k: _dev[k] / T for k in _dev} for _dev in devs]
        sortby = 'Feature'
        I_pprof = pd.DataFrame(imp)\
                    .unstack(1)\
                    .reset_index()\
                    .rename(columns={'level_0': sortby,
                                     'level_1': 'Model',
                                     0: colname})
        I_pprof[sortby] = I_pprof[sortby].astype('category')
        I_pprof[sortby].cat.set_categories(I.index, inplace=True)
        I_pprof.sort(sortby, inplace=True)

        return I, I_pprof

def plot_imp(I, colname="Importance"):
    """Plot overall feature importance."""
    sns.plt.figure()
    deep_blue = sns.color_palette('colorblind')[0]
    ax = sns.barplot(data=I, x=colname, y=I.index, color=deep_blue)
    ax.set(title='Feature Importance for Grade Prediction',
           ylabel='Feature',
           xlabel='Proportion of Deviation From Intercept')
    ax.figure.show()
    return ax

def plot_pprof_imp(I_pprof, colname="Importance", sortby="Feature"):
    """Plot per-profile feature importance."""
    sns.plt.figure()
    ax = sns.barplot(data=I_pprof, x=colname, y=sortby, hue='Model')
    ax.set(title='Feature Importance Per Model', xlabel=colname)
    ax.figure.show()
    return ax


def make_parser():
    parser = argparse.ArgumentParser(
        description="mixed-membership multi-linear regression")
    parser.add_argument(
        '-m', '--method',
        choices=('sgd', 'als'), default='sgd',
        help='learning method to use')
    parser.add_argument(
        '-tr', '--train',
        help='path of training data file in csv format')
    parser.add_argument(
        '-te', '--test',
        help='path of test data file in csv format')
    parser.add_argument(
        '-k', '--nmodels',
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
    parser.add_argument(
        '--bounds', default='0,4',
        help='upper,lower bound for rating bounding')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-i', '--iters',
        type=int, default=10,
        help='number of iterations')
    parser.add_argument(
        '-e', '--epsilon',
        type=float, default=0.0001,
        help='stopping threshold for early stopping test')
    parser.add_argument(
        '-s', '--init-std',
        type=float, default=(1. / 2 ** 4),
        help='standard deviation of Gaussian noise used in model param'
             'initialization')
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
    parser.add_argument(
        '-f', '--feature-guide',
        default='',
        help='file to specify target, categorical, and real-valued features; '
             'see the docstring for more detailed info on the format')
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


    # Sanity check bounds argument.
    bounds = args.bounds.split(',')
    if len(bounds) != 2:
        print 'bounds must be comma-separated, got %s' % args.bounds
        sys.exit(BOUNDS_FORMAT)


    logging.info('reading train/test files')
    try:
        data_and_names = \
            read_train_test(args.train, args.test, args.feature_guide)
        train = data_and_names[0]
        eids, X, y, test_eids, test_X, test_y, f_indices, nb = \
            preprocess(*data_and_names)
        errno = None
    except IOError as e:
        errno = BAD_FILENAME
    except BadFeatureConfig as e:
        errno = BAD_FEATURE_CONF

    if errno is not None:
        logging.error(e.message)
        sys.exit(errno)


    # Init model using cmdline args.
    model = IPR(nmodels=args.nmodels,
                lambda_w=args.lambda_w,
                lambda_b=args.lambda_b,
                iters=args.iters,
                init_std=args.init_std,
                nonneg=args.nonneg,
                verbose=args.verbose,
                lrate=args.lrate,
                epsilon=args.epsilon)

    # Train IPR model.
    model.fit(X, y, eids, nb)


    # Make predictions and evaluate in terms of RMSE.
    predictions = model.predict(test_X, test_eids, nb)
    err = test_y - predictions
    print 'IPR RMSE:\t%.4f' % rmse_from_err(err)

    baseline_pred = np.random.uniform(0, 4, len(test_y))
    print 'UR RMSE:\t%.4f' % rmse_from_err(baseline_pred - test_y)

    gm_pred = np.repeat(y.mean(), len(test_y))
    print 'GM RMSE:\t%.4f' % rmse_from_err(gm_pred - test_y)


    # Save model params.
    if args.output:
        try:
            model.save(args.output)
        except OSError as err:
            logging.error("model save failed: %s" % str(err))


    # Calculate feature importance metrics.
    I, I_pprof = model.feature_importance(
        X, y, eids, nb, train, f_indices, args.feature_guide)

    ax1 = plot_imp(I)
    ax2 = plot_pprof_imp(I_pprof)

    raw_input()
