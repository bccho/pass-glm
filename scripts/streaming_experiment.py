### Script for testing streaming PASS for logistic regression ###

from __future__ import print_function

import os.path
import sys
import argparse
import time
import cPickle as cpk

import numpy as np
import numpy.random as npr
import numpy.core.numeric as _nx
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
from passglm.bayes_logistic import StreamingLogisticRegression
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
    parser.add_argument('--name', default=None)
    parser.add_argument('-n', '--num-test-files', type=int, default=-1)
    parser.add_argument('-d', '--max-dimension', type=int, default=0)
    parser.add_argument('-r', '--learning-rate', type=float, default=0)
    parser.add_argument('-t', '--target-dim', type=int, default=0)
    parser.add_argument('--do-not-save-cov', action='store_true')
    parser.add_argument('--num-cpus', type=int, default=1)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('-R', type=float, default=3.0)
    parser.add_argument('--include-offset', action='store_true',
                        help='add dummy feature equal to 1')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--sparse-random-projection', action='store_true')
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
    results_file = out_dir + '/results.cpk'
    if os.path.exists(results_file):
        print('loading existing results...')
        with file(results_file, 'r') as f:
            results = cpk.load(f)
    else:
        results = run_experiment(args)
        if args.do_not_save_cov:
            del results['pass']['cov']
        with file(results_file, 'w') as f:
            cpk.dump(results, f)

    sns.set_style('white')
    sns.set_context('notebook', font_scale=2, rc={'lines.linewidth': 2})
    plot_roc_curves(results, 'ROC curves', out_dir)
    #plot_all_results(results, out_dir)

    with open(out_dir + '/TLLs.txt', 'w') as f:
        print('SGD TLL =', results['sgd']['TLL'], file=f)
        print('PASS TLL =', results['pass']['TLL'], file=f)


def results_dir_name(args):
    if args.name is None:
        file_names = '-'.join([os.path.basename(f).split('.')[0] for f in args.data_file_path])
    else:
        file_names = args.name
    pca_str = '-pca' if args.pca else ''
    srp_str = '-srp' if args.sparse_random_projection else ''
    pca_str += srp_str
    #radius_str = '-'.join(['%.2f' % r for r in args.R])
    return '%s%s-d-%d-R-%.2f-eta-%.4f-discount-%.4f' % (
        file_names, pca_str, args.max_dimension, args.R,
        args.learning_rate, args.discount)


def run_experiment(args):
    include_svb = False
    # split data into train and test sets
    train_files = args.data_file_path
    if args.num_test_files == -1:
        num_test_files = int(len(train_files)*.2+1)
    else:
        num_test_files = args.num_test_files
    test_files = train_files[-num_test_files:]
    del train_files[-num_test_files:]
    print('train files:', train_files)
    print('test files:', test_files)

    results = { 'args' : args,
                'pass' : dict(),
                'sgd'  : dict(),
#                'svb'  : dict()
              }
    # init models
    if args.learning_rate > 0:
        print('using constant learning rate')
        learning_rate = 'constant'
        eta0 = args.learning_rate
    else:
        print('using "optimal" learning rate')
        learning_rate = 'optimal'
        eta0 = 0
    sgd_model = SGDClassifier(loss='log',
                              fit_intercept=False,
                              #alpha=0.5/PRIOR_VAR,
                              learning_rate=learning_rate,
                              eta0=eta0)
    ray.init(num_cpus=args.num_cpus)
    pass_model = PASSClassifier(r=args.R, alpha=0.5/PRIOR_VAR,
                                discount=args.discount)
    svb_model = StreamingLogisticRegression(alpha=1/PRIOR_VAR,
                                            fit_intercept=False,
                                            n_iter_solver=5)
    # train models
    pp = True
    pca_obj = None
    srp_obj = None
    for train_file in train_files:
        Xs, ys, pp, pca_obj, srp_obj = load_train_or_test_data(
            train_file, args, pp=pp, pca_obj=pca_obj, srp_obj=srp_obj)
        print('running SGD...')
        with Timer('SGD'):
            for i, (X, y) in enumerate(zip(Xs, ys)):
                print('split', i)
                # print(np.mean(X), np.mean(X**2)*X.shape[1])
                sgd_model.partial_fit(X, y, classes=[-1, 1])
                # print('running SVB')
                # with Timer('SVB'):
                #     svb_model.partial_fit(X, y)
        print('running PASS...')
        with Timer('PASS'):
            pass_model.distributed_partial_fits(Xs, ys, args.num_cpus, args.num_cpus+1)

    # record model parameters
    results['sgd']['mean'] = sgd_model.coef_
    results['pass']['mean'], results['pass']['cov'] = pass_model.mean_cov()
    # results['svb']['mean'] = svb_model.coef_
    # results['svb']['cov'] = np.diag(svb_model.sigma_[0])
    # run on test data

    model_tlls = dict()
    model_lls = dict()
    # for model in ['sgd', 'pass', 'svb']:
    for model in ['sgd', 'pass']:
        model_tlls[model] = []
        model_lls[model] = np.array([])
    weights = []
    y_test_all = np.array([])
    for test_file in test_files:
        X_tests, y_tests, pp, pca_obj, srp_obj = load_train_or_test_data(
            test_file, args, pp=pp, pca_obj=pca_obj, srp_obj=srp_obj,
            for_test=True)
        for X_test, y_test in zip(X_tests, y_tests):
            if sp.issparse(X_test):
                X_test = sp.diags(y_test).dot(X_test)
            else:
                X_test = y_test[:, np.newaxis] * X_test
            tll, lls = test_log_likelihood(X_test, results['sgd']['mean'])
            model_tlls['sgd'].append(tll)
            model_lls['sgd'] = np.hstack([model_lls['sgd'], lls - 1e-12])
            # for model in ['svb', 'pass']:
            for model in ['pass']:
                r = results[model]
                tll, lls = gaussian_test_log_likelihood(X_test, r['mean'], r['cov'])
                model_tlls[model].append(tll)
                model_lls[model] = np.hstack([model_lls[model], lls])
            weights.append(X_test.shape[0])
            y_test_all = np.hstack([y_test_all, y_test])
            X_test = None

    # for model in ['sgd', 'pass', 'svb']:
    for model in ['sgd', 'pass']:
        r = results[model]
        r['TLL'] = np.average(model_tlls[model], weights=weights)
        r['fpr'], r['tpr'], r['roc_auc'] = calculate_roc_auc(model_lls[model],
                                                             y_test_all)
        model_name = model.upper()
        print(model_name, 'TLL =', r['TLL'])
        print(model_name, 'ROC AUC =', r['roc_auc'])
    # r = results['pass']
    # r['TLL'] = np.average(pass_tlls, weights=weights)
    # r['fpr'], r['tpr'], r['roc_auc'] = calculate_roc_auc(pass_lls, y_test_all)
    # print('PASS TLL =', r['TLL'])
    # print('PASS ROC AUC =', r['roc_auc'])
    return results


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
    split_size = 1e4 if for_test else 2e4
    num_splits = max(1, int(X.shape[0] / split_size + .5))
    if num_splits > 1:
        print('num splits =', num_splits)
    Xs = sparse_vsplit(X, num_splits)
    ys = sparse_vsplit(y, num_splits)
    return Xs, ys, pp, pca_obj, srp_obj


def gaussian_test_log_likelihood(Z_test, mean, cov, num_samples=100):
    #theta_samples = npr.multivariate_normal(mean, cov, num_samples)
    theta_samples = multivariate_gaussian(mean, cov, num_samples)
    return test_log_likelihood(Z_test, theta_samples.T)


def multivariate_gaussian(mean, cov, samples=1):
    chol = np.linalg.cholesky(cov)
    mean = mean.reshape((-1,1))
    samples = mean + chol.dot(npr.randn(mean.size, samples))
    return samples


def test_log_likelihood(Z_test, theta_samples):
    ll = logistic_likelihood(theta_samples.T, Z_test, sum_result=False)
    print('ll shape =', ll.shape, '#Z_test =', Z_test.shape[0],
          '# theta samples =', theta_samples.shape[0])
    return (logsumexp(ll, b=1.0/ll.size),
            logsumexp(ll, axis=1, b=1.0/ll.shape[1]))


def calculate_roc_auc(ll_test, y_test):
    scores = ll_test.copy()
    neg_inds = y_test == -1
    scores[neg_inds] = np.log1p(-np.exp(ll_test[neg_inds]))
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curves(results, name, out_dir):
    plt.figure()
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k', linestyle='--')
    for key in ['pass', 'sgd', 'svb']:
        if key in results:
            r = results[key]
            if key == 'pass':
                key = 'pass-lr2'
            plt.plot(r['fpr'], r['tpr'],
                     label='%s (area = %0.3f)' % (key.upper(), r['roc_auc']))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    sns.despine()
    plt.savefig('%s/%s.pdf' % (out_dir, name.replace(' ', '-')),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
