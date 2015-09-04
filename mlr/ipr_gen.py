"""
Generate data according to the Individualized Personalized Regression Model.

"""
import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd


# Error codes
BOUNDS_FORMAT = 1000


def make_parser():
    parser = argparse.ArgumentParser(
        description="mixed-membership multi-linear regression")
    parser.add_argument(
        '-o', '--output',
        default='',
        help='file to save generated data to; defualt name uses params and a'
             'timestamp')
    parser.add_argument(
        '-e', '--nentities',
        type=int, default=2,
        help='number of entities to use in data generation')
    parser.add_argument(
        '-nc', '--ncats',
        type=int, default=2,
        help='number of categorical features to use in data generation')
    parser.add_argument(
        '-nr', '--nreals',
        type=int, default=8,
        help='number of real-valued features to use in data generation')
    parser.add_argument(
        '-t', '--target',
        default='grade',
        help='name to give to target variable')
    parser.add_argument(
        '-b', '--bounds',
        help='comma-separated target range bounds')
    parser.add_argument(
        '-k', '--nmodels',
        type=int, default=3,
        help='number of linear regression models')
    parser.add_argument(
        '-n', '--nonneg',
        action='store_true', default=False,
        help='enable non-negativity constraints on all params')
    parser.add_argument(
        '-v', '--verbose',
        type=int, default=0, choices=(0, 1, 2),
        help='verbosity level; 0=None, 1=INFO, 2=DEBUG')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

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

    try:
        lo, hi = map(float, bounds)
    except ValueError:
        print 'bounds not representable as floats, got %s' % args.bounds
        sys.exit(BOUNDS_FORMAT)


