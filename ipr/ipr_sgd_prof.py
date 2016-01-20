import pstats, cProfile
from ipr import read_data
from cipr import fit_ipr_sgd


if __name__ == "__main__":
    data = '../data/data-n500-m50-t4-d5538'
    train = '%s-%s' % (data, 'train.csv')
    test = '%s-%s' % (data, 'test.csv')
    fguide = 'fguide-ipr.conf'

    eids, X, y, test_eids, test_X, test_y, f_indices, nb = \
        read_data(train, test, fguide)

    outname = 'IPR_SGD-profile.prof'
    cProfile.runctx('''fit_ipr_sgd(
        X, y, eids, nb,
        k=3,
        lambda_w=0.1,
        lambda_b=0.0,
        iters=50,
        std=0.01,
        nn=0,
        verbose=0,
        lrate=0.001,
        eps=0.00001)''',
        globals(), locals(), outname)

    s = pstats.Stats(outname)
    s.strip_dirs().sort_stats("cumtime").print_stats()


