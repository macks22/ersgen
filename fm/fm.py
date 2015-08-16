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
import sys
import logging
import argparse

import numpy as np
import scipy as sp
import pandas as pd
from sklearn import preprocessing


# Error codes
BOUNDS_FORMAT = 1000
MISSING_ATTRIBUTE = 1001
DIM_MISMATCH = 1002
BAD_FEATURE_CONF = 1003
BAD_FILENAME = 1004


class BadFeatureConfig(Exception):
    """Raise when bad feature configuration file is found."""
    pass


def read_feature_guide(fname):
    """Read the feature guide and parse out the specification.

    The expected file format is the following:

        t:<target>;
        c:<comma-separated categorical variable names>;
        r:<comma-separated real-valued variable names>;

    Whitespace is ignored, as are lines that start with a "#" symbol. Any
    variables not included in one of the three groups is ignored.

    Args:
        fname (str): Path of the file containing the feature guide.

    Returns:
        iterable: (target, categoricals list, real-valued list).

    """
    with open(fname) as f:
        lines = [l.strip() for l in f.read().split('\n')
                 if not l.startswith('#') and l.strip()]

    parsing = False
    keys = ['t', 'c', 'r']
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

    num_features = len(vars['c']) + len(vars['r'])
    if not num_features > 0:
        raise BadFeatureConfig('no predictors specified')

    spec = (vars['t'][0], vars['c'], vars['r'])
    target, cats, reals = spec
    logging.info('read the following feature guide:')
    logging.info('target: %s' % target)
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
    else:
        data[key] = data[key].apply(lambda _id: id_map[_id])
    return id_map


def read_data(train_file, test_file, conf_file):
    """Read the train and test data according to the feature guide in the
    configuration file. Return the train and test data as (X, y) pairs for the
    train and test data.

    Args:
        train_file (str): Path of the CSV train data file.
        test_file (str): Path of the CSV test data file.
        conf_file (str): Path of the configuration file.

    Returns:
        tuple: (train_X, train_y, test_X, test_y), preprocessed train/test data.

    """
    try:
        target, cats, reals = read_feature_guide(conf_file)
    except IOError:
        raise IOError('invalid feature guide conf file path: %s' % conf_file)

    to_read = [target] + cats + reals
    def read_file(name, fpath):
        try:
            data = pd.read_csv(fpath, usecols=to_read)
        except IOError:
            raise IOError('invalid %s file path: %s' % (name, fpath))
        except ValueError as err:
            attr_name = err.args[0].split("'")[1]
            attr_type = ('categorical' if attr_name in cats else
                         'real-valued' if attr_name in reals else
                         'target')
            raise BadFeatureConfig('%s attribute %s not found in %s file' % (
                attr_type, attr_name, name))

        return data

    train = read_file('train', train_file)
    test = read_file('test', test_file)
    nd_train = train.shape[0]
    nd_test = test.shape[0]
    logging.info('number of dyads: train=%d, test=%d' % (nd_train, nd_test))

    # Separate X, y for train/test data.
    train_y = train[target].values
    test_y = test[target].values

    # Z-score scaling of real-valued features.
    scaler = preprocessing.StandardScaler()
    train_reals = scaler.fit_transform(train[reals])
    test_reals = scaler.transform(test[reals])

    # One-hot encoding of categorical features.
    all_cats = pd.concat((train[cats], test[cats]))
    encoder = preprocessing.OneHotEncoder()
    enc_cats = encoder.fit_transform(all_cats)
    train_cats = enc_cats[:nd_train]
    test_cats = enc_cats[nd_train:]

    logging.info('after one-hot encoding, found # unique values:')
    for attr, n_values in zip(cats, encoder.n_values_):
        logging.info('%s: %d' % (attr, n_values))

    # Put all features together.
    train_X = sp.sparse.hstack((train_cats, train_reals))
    test_X = sp.sparse.hstack((test_cats, test_reals))
    logging.info('Total of %d features after encoding' % train_X.shape[1])

    return train_X, train_y, test_X, test_y


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
        '-k', '--dim',
        type=int, default=8,
        help='dimensionality of factorized interactions')
    parser.add_argument(
        '-i', '--iterations',
        type=int, default=50,
        help='number of iterations to run for')
    parser.add_argument(
        '--bounds', default='0,4',
        help='upper,lower bound for rating bounding')
    parser.add_argument(
        '-std', '--init-stdev',
        type=float, default=0.01,
        help='standard deviation of Gaussian noise initialization for'
             'factorized terms')
    parser.add_argument(
        '-s', '--stopping-threshold',
        type=float, default=0.00001,
        help='early stopping threshold')
    parser.add_argument(
        '-f', '--feature-guide',
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
        train_X, train_y, test_X, test_y = read_data(
            args.train, args.test, args.feature_guide)
        errno = None
    except IOError as e:
        errno = BAD_FILENAME
    except BadFeatureConfig as e:
        errno = BAD_FEATURE_CONF

    if errno is not None:
        logging.error(e.message)
        sys.exit(errno)

    # We have the data, let's begin.
