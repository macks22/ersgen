import argparse
import logging

import numpy as np

from pmf import fit_pmf
from util import read_data


def make_parser():
    parser = argparse.ArgumentParser(
        'Probabilistic Matrix Factorization')
    parser.add_argument(
        '-d', '--dim',
        type=int, default=10,
        help='latent feature vector dimensionality')
    parser.add_argument(
        '-i', '--iters',
        type=int, default=50,
        help='number of iterations to learn for')
    parser.add_argument(
        '-l', '--lambda_',
        type=float, default=0.01,
        help='regularization term')
    parser.add_argument(
        '-nb', '--nbatches',
        type=int, default=9,
        help='number of batches per epoch')
    parser.add_argument(
        '-bs', '--batch_size',
        type=int, default=100000,
        help='number of records to process per batch')
    parser.add_argument(
        '-v', '--verbose',
        type=int, default=0, choices=(0, 1, 2),
        help='enable verbose output logging')
    return parser


if __name__ == "__main__":
    np.random.seed(1234)
    parser = make_parser()
    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose == 2 else
               logging.INFO if args.verbose == 1 else
               logging.ERROR),
        format="[%(asctime)s]: %(message)s")

    # Read data.
    logging.info('reading train/test data')
    header_names, train, probe = read_data()
    uid, iid, target = header_names

    # Fit model.
    logging.info('fitting PMF model')
    w_u, w_i = fit_pmf(train, probe, uid, iid, target, nf=args.dim,
                       max_epoch=args.iters, lambda_=args.lambda_,
                       nbatches=args.nbatches, N=args.batch_size)

    # Finally, save weights at the end.
    logging.info('saving data')
    np.savetxt('w_u.csv', w_u, delimiter=',')
    np.savetxt('w_i.csv', w_i, delimiter=',')
