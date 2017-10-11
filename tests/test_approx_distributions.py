from __future__ import absolute_import

import numpy as np
import numpy.random as npr
from numpy.testing import *

from numpy.polynomial import Polynomial

import passglm.approx_distributions as adists
import passglm.distributions as dists

npr.seed(10)

def chebyshev_approximation_test():
    test_funs = [(lambda x: np.log1p(np.exp(-x)), 'log sigmoid'),
                 (lambda x: x**4, 'x^4'),
                 (lambda x: np.exp(3*x), 'exp(3x)'),
                 (lambda x: np.sin(4*x), 'sin(4x)'),
                 (lambda x: np.cos(x), 'cos(x)'),
                 (lambda x: np.cos(x**2 - x), 'cos(x^2 - x)'),
                 ]
    test_locs = np.linspace(-1,1,100)
    for fun, fun_name in test_funs:
        _check_chebyshev_approximation(fun, fun_name, test_locs)


def _check_chebyshev_approximation(fun, fun_name, test_locs):
    p = Polynomial(adists.chebyshev_approximation(fun, degree=10))
    vals = fun(test_locs)
    approx_vals = p(test_locs)
    assert_allclose(vals, approx_vals, atol=1e-4,
                    err_msg='failed on %s' % fun_name)


def moment_approximation_test():
    test_funs = [(lambda x: np.log1p(np.exp(-x)), 'log sigmoid'),
                 (lambda x: x**4, 'x^4'),
                 (lambda x: np.cos(x), 'cos(x)'),
                 ]

    for N in [1, 5, 100, 1000]:
        for dim in [5]:
            for fun, fun_name in test_funs:
                _check_moment_approximation(fun, fun_name, N, 6, dim)


def _check_moment_approximation(fun, fun_name, N, degree, dim):
    data = np.random.rand(N, dim) / np.sqrt(dim)

    coeffs = adists.chebyshev_approximation(fun, degree,
                                            lambda x: x > 1 and x % 2 == 1)
    moments = adists.Moments(dim, degree, coeffs)
    moments.add_all(data)
    for i in range(100):
        theta = np.random.rand(dim) / np.sqrt(dim)
        approx_val = moments.approximate_value_at(theta)
        val = np.sum(fun(data.dot(theta)))
        assert_almost_equal(val, approx_val, decimal=3,
                            err_msg='failed on %s, N=%d, dim=%d'
                                    % (fun_name, N, dim))


def make_approx_likelihood_fun_test():
    test_funs = [(lambda x: np.log1p(np.exp(-x)), 'log sigmoid'),
                 (lambda x: x**4, 'x^4'),
                 (lambda x: np.cos(x), 'cos(x)'),
                 ]

    for N in [1, 5, 100, 1000]:
        for dim in [5]:
            for fun, fun_name in test_funs:
                _check_approx_likelihood(fun, fun_name, N, 8, dim)


def _check_approx_likelihood(fun, fun_name, N, degree, dim):
    data = np.random.rand(N, dim) / np.sqrt(dim)

    (approx_fun,) = adists.make_approx_likelihood_fun(
                            fun, data, degree, lambda x: x > 1 and x % 2 == 1)
    for i in range(100):
        theta = np.random.rand(dim) / np.sqrt(dim)
        approx_val = approx_fun(theta)
        val = np.sum(fun(data.dot(theta)))
        assert_almost_equal(val, approx_val, decimal=3,
                            err_msg='failed on %s, N=%d, dim=%d'
                                    % (fun_name, N, dim))
