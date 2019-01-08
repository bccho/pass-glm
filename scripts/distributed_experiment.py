### Script for testing parallelization of PASS for logistic regression ###
# Rerun in sequence with differing arguments to -n

from __future__ import print_function

import os.path
import sys
import argparse
import time
import pickle

import numpy as np
import numpy.random as npr
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.misc import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn import preprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import ray

from passglm.data import load_data
from passglm.approx_distributions import PASSClassifier
from passglm.inference import mh, gelman_rubin_diagonostic
from passglm.distributions import (logistic_likelihood,
                                    logistic_likelihood_grad,
                                    logistic_likelihood_hessian)
import passglm.evaluation as ev
from passglm.utils import create_folder_if_not_exist, Timer, sparse_vsplit

PRIOR_VAR = 4.0
#PRIOR_VAR = .5 / 0.0001

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file_path', nargs='+')
    parser.add_argument('-a', '--address', default=None)
    parser.add_argument('-n', '--num-cpus', type=int, required=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--name', default=None)
    parser.add_argument('-d', '--max-dimension', type=int, default=0)
    parser.add_argument('-t', '--target-dim', type=int, default=0)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('-R', type=float, default=3.0)
    parser.add_argument('--include-offset', action='store_true',
                        help='add dummy feature equal to 1')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--sparse-random-projection', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    np.set_printoptions(precision=5, suppress=True)
    args = parse_arguments()
    if args.target_dim <= 0:
        args.target_dim = None
    if args.sparse_random_projection and args.pca:
        sys.exit("can't use pca and sparse random projection")

    out_dir = 'results/' + results_dir_name(args)
    create_folder_if_not_exist(out_dir)
    results_file = out_dir + '/results.pickle'
    if os.path.exists(results_file):
        print('loading existing results...')
        with file(results_file, 'r') as f:
            results = pickle.load(f)
        if args.force and args.num_cpus in results['ns']:
            i = results['ns'].index(args.num_cpus)
            results['ns'].pop(i)
            results['times'].pop(i)
    else:
        results = None
    if results is None or args.num_cpus not in results['ns']:
        results = run_experiment(args, results)
        with file(results_file, 'w') as f:
            pickle.dump(results, f)
    sns.set_style('white')
    sns.set_context('notebook', font_scale=3, rc={'lines.linewidth': 3})
    plot_timing(results, 'times', out_dir)
    plot_speedup(results, 'speedup', out_dir)


def results_dir_name(args):
    if args.name is None:
        file_names = '-'.join([os.path.basename(f).split('.')[0] for f in args.data_file_path])
    else:
        file_names = args.name
    pca_str = '-pca' if args.pca else ''
    srp_str = '-srp' if args.sparse_random_projection else ''
    pca_str += srp_str
    #radius_str = '-'.join(['%.2f' % r for r in args.R])
    return 'distributed-%s%s-d-%d-R-%.2f-discount-%.4f' % (
        file_names, pca_str, args.max_dimension, args.R, args.discount)


def run_experiment(args, results=None):
    # split data into train and test sets
    train_files = args.data_file_path
    print('train files:', train_files)
    if results is None:
        results = { 'args' : args,
                    'ns' : [],
                    'times' : list(),
                  }
    # init models
    n = args.num_cpus
    # if n > 1:
    if args.address is not None:
        ray.init(redis_address=args.address)
    else:
        ray.init(num_cpus=n)
    model = PASSClassifier(r=args.R, alpha=0.5/PRIOR_VAR,
                           discount=args.discount)
    # distributed_model = PASSClassifier(r=args.R, alpha=0.5/PRIOR_VAR,
    #                             discount=args.discount, distributed=True)
    # load training data
    pp = True
    pca_obj = None
    srp_obj = None
    all_Xs = []
    all_ys = []
    for train_file in train_files:
        Xs, ys, pp, pca_obj, srp_obj = load_train_or_test_data(
            train_file, args, pp=pp, pca_obj=pca_obj, srp_obj=srp_obj)
        all_Xs.extend(Xs)
        all_ys.extend(ys)

    min_num = 1 if args.full else n
    while True:
        if n not in results['ns']:
            # time training of models
            print('running with n =', n)
            with Timer('n = %d' % n) as t:
                # if n == 1:
                #     cnt = 0
                #     for i, (X, y) in enumerate(zip(all_Xs, all_ys)):
                #         print('split', i)
                #         if sp.issparse(X):
                #             X = sp.diags(y).dot(X)
                #         else:
                #             X = y[:, np.newaxis] * X
                #         if n > 1:
                #             model.distributed_partial_fit(X)
                #             cnt += 1
                #         else:
                #             model.partial_fit(X)
                #         if n > 1 and cnt >= n*2:
                #             print('added', model.collect_distributed_fits(), 'splits')
                #             cnt = 0
                #     if n > 1:
                #         model.collect_distributed_fits()
                # else:
                model.distributed_partial_fits(all_Xs, all_ys, n, n+1)
            results['ns'].append(n)
            results['times'].append(t.interval)
        new_n = n / 2
        if n <= min_num:
            break
        [do_nothing.remote() for i in range(n - new_n)]
        n = new_n
    return results


@ray.remote
def do_nothing():
    while True:
        time.sleep(100000)


def load_train_or_test_data(data_file, args, pp=True, pca_obj=None,
                            srp_obj=None, for_test=False):
    print('loading %s...' % data_file)
    if args.pca or args.sparse_random_projection:
        max_dim = 0
        include_offset = False
    else:
        max_dim = args.max_dimension
        include_offset = args.include_offset
    X, y, pp = load_data(data_file,
                         data_file.split('.')[-1],
                         max_dim=max_dim,
                         preprocess=pp,
                         include_offset=include_offset,
                         target_dim=args.target_dim)
    if args.pca and args.max_dimension > 0:
        print('performing PCA')
        if pca_obj is None:
            pca_comps = args.max_dimension
            if args.include_offset:
                pca_comps -= 1
            pca_obj = PCA(n_components=pca_comps).fit(X)
        X = pca_obj.transform(X)
        if args.include_offset:
            X = preprocessing.add_dummy_feature(X)
    if args.sparse_random_projection:
        print('performing sparse random projection')
        if srp_obj is None:
            if args.max_dimension > 0:
                n_components = args.max_dimension
                print(n_components * 10, X.shape[1])
                dense_output = n_components * 10 < X.shape[1]
            else:
                n_components = 'auto'
                dense_output = True  # not sure if this is a good idea...
            srp_obj = SparseRandomProjection(n_components=n_components,
                                             dense_output=dense_output,
                                             random_state=0).fit(X)
        X = srp_obj.transform(X)
        if args.include_offset:
            X = preprocessing.add_dummy_feature(X)
        if sp.issparse(X) and (X.nnz > np.prod(X.shape) / 3.0 or
                               X.shape[1] <= 20):
            print("X is either low-dimensional or not very sparse, so "
                  "converting to a numpy array")
            X = X.toarray()
    # Z = sp.diags(y).dot(X)
    num_features = X.shape[1]
    print('%d total training data points of dimension %d' % X.shape)
    if sp.issparse(X):
        print('density =', float(X.nnz) / np.prod(X.shape))
    # split data further, if necessary
    split_size = 1e4 if for_test else 2e5
    num_splits = max(1, int(X.shape[0] / split_size + .5))
    if num_splits > 1:
        print('num splits =', num_splits)
    Xs = sparse_vsplit(X, num_splits)
    ys = sparse_vsplit(y, num_splits)
    return Xs, ys, pp, pca_obj, srp_obj



def plot_timing(results, name, out_dir):
    plt.figure()
    plt.clf()
    indices = np.argsort(results['ns'])
    ns = np.array(results['ns'])[indices]
    times = np.array(results['times'])[indices]
    plt.plot(np.log2(ns), np.log2(times))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('cores')
    plt.ylabel('time (sec)')
    sns.despine()
    ax = plt.gca()
    round_to_n = lambda x, n: str(round(x, -int(np.floor(np.log10(x))) + (n - 1)))
    ax.set_xticklabels([str(int(2**v)) for v in ax.get_xticks()])
    ax.set_yticklabels([round_to_n(2**v, 2) for v in ax.get_yticks()])
    plt.savefig('%s/%s.pdf' % (out_dir, name.replace(' ', '-')),
                bbox_inches='tight')
    plt.close()


def plot_speedup(results, name, out_dir):
    plt.figure()
    plt.clf()
    indices = np.argsort(results['ns'])
    ns = np.array(results['ns'])[indices]
    times = np.array(results['times'])[indices]
    speedup = np.max(times) / np.array(results['times'])[indices]
    plt.plot(ns, speedup)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('cores')
    plt.ylabel('speedup')
    sns.despine()
    plt.savefig('%s/%s.pdf' % (out_dir, name.replace(' ', '-')),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
