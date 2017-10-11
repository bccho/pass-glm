# Author: Jonathan Huggins <jhuggins@mit.edu>

from __future__ import absolute_import, print_function

import numbers
from array import array
import time
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.testing import *
import scipy.sparse as sp
import scipy.misc as spmisc
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.misc import logsumexp

import ray
from .utils import Timer
from .distributions import (logistic_likelihood,
                            logistic_likelihood_grad,
                            logistic_likelihood_hessian)
from . import _approx_distributions as _ad
from .data import generate_binary_data

# from .profiling import line_profiled
# PROFILING = False

class PASSClassifier(object):
    """PASS-GLM classifier

    Supports distributed fitting using ray
    """
    def __init__(self, r=3.0, alpha=.1, loss='log', discount=1.0):
        self.r = r
        self.alpha = alpha
        if loss == 'log':
            self.link = lambda x: -np.log1p(np.exp(-r * x))
        else:
            raise ValueError('Unsupported loss "%s"' % loss)
        if discount > 1.0 or discount <= 0.0:
            raise ValueError('discount rate must be in (0, 1]')
        self.discount = discount
        self.first_partial_fit_complete = False
        self.moment_updates = []

    def _init_model(self, Z):
        coeffs = chebyshev_approximation(self.link, 2)
        self.moments = Moments2(Z.shape[1], coeffs, sparse=sp.issparse(Z))

    def fit(self, Z):
        """Fit PASS-GLM model

        Parameters
        ----------
        Z : array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        self : object
        """
        self._init_model(Z)
        self.moments.add_all(Z)
        self.first_partial_fit_complete = False
        return self

    def partial_fit(self, Z):
        """Incrementally fit PASS-GLM model

        Parameters
        ----------
        Z : array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        self : object
        """
        if not self.first_partial_fit_complete:
            self._init_model(Z)
            self.first_partial_fit_complete = True
        elif self.discount < 1:
            self.moments.moments *= self.discount
        self.moments.add_all(Z)
        return self

    def distributed_partial_fits(self, Xs, ys, n_min, n_max):
        """Incrementally fit PASS-GLM model using ray.

        Parameters
        ----------
        Xs : list of array-like matrices, shape=(n_samples_i, n_features)

        ys : list of array-like, shape=(n_samples_i,)

        n_min : int
            return after all but n_min data chunks have been processed

        n_max : int
            maximum number of chunks to distribute at once
        """
        for i, (X, y) in enumerate(zip(Xs, ys)):
            print('batch', i)
            self._distributed_partial_fit(X, y)
            if len(self.moment_updates) >= n_max:
                print('added', self._collect_distributed_fits(n_min), 'splits')
        self._collect_distributed_fits()

    def _distributed_partial_fit(self, Z, y=None):
        if not self.first_partial_fit_complete:
            self._init_model(Z)
            self.first_partial_fit_complete = True
        self.moment_updates.append(
            calculate_x_moments.remote(self.moments.coeffs[0],
                                       Z,
                                       self.moments.sparse,
                                       ys=y))

    def _collect_distributed_fits(self, n_min=0):
        n_min = max(0, n_min)
        moment_updates = self.moment_updates
        num_moments = len(moment_updates)
        while len(moment_updates) > n_min:
            errors = ray.error_info()
            if len(errors) > 0:
                print('errors:', errors)
            print('n min =', n_min, 'remaining =', len(moment_updates), 'object ids =', moment_updates)
            ready_id, moment_updates = ray.wait(moment_updates, num_returns=1)
            print('processing', ready_id[0])
            update = ray.get(ready_id[0])
            self.moments.moments += update
        self.moment_updates = moment_updates
        return num_moments - len(moment_updates)

    def mean_cov(self):
        """
        Get the mean and covariance of the approximate PASS-GLM posterior

        Returns
        -------
        mean : ndarray of floats with shape=(n_features,)
        cov : ndarray of floats with shape=(n_features, n_features)
        """
        mean, cov, _ = self.moments.mean_cov(0.5 / self.alpha)
        return self.r * mean, self.r**2 * cov


class LaplaceClassifier(object):
    def __init__(self, coeffs, intercept, alpha=.0001, num_samples=100):
        self.coeffs = coeffs
        self.intercept = intercept
        self.alpha = alpha
        self.num_samples = num_samples

    def fit(X, y):
        _, d = X.shape
        ident = sp.eye(d) if sp.issparse(X) else np.eye(d)
        hess = logistic_likelihood_hessian(self.coeffs, sp.diags(y).dot(X)) + alpha * ident
        self.cov = np.linalg.inv(hess.toarray())
        self.samples = multivariate_gaussian(self.coeffs, self.cov, self.num_samples).T

    def decision_function(X_test):
        ll = -np.log1p(np.exp(-self.intercept - X_test.dot(self.samples)))
        return logsumexp(ll, axis=1, b=1.0/ll.shape[1])


def multivariate_gaussian(mean, cov, samples=1):
    chol = np.linalg.cholesky(cov)
    mean = mean.reshape((-1,1))
    samples = mean + chol.dot(npr.randn(mean.size, samples))
    return samples


def make_approx_likelihood_fun(lik_fun, data, degree, is_zero=None,
                               timeit=False, verbose=False,
                               use_deg2_special=False, prior_var=None):
    """
    Construct an approximate GLM log-likelihood mapping function

    Parameters
    ----------
    lik_fun : function
        The exact log-likelihood mapping function

    data : array-like matrix, shape=(n_samples, n_features)

    degree : int
        Degree of the Chebyshev approximation (must be greater than 0)

    is_zero : int iterable, optional
        If given, do not calculate coefficients of the given degrees and instead
        assume these coefficients are zero.

    timeit : boolean or string, optional
        If True or 'all', or 'moments' then times how long it takes to
        construct the approximation. If 'moments' then only time how long
        it takes to calculate the momoments of the data.

    verbose : boolean, optional
        Default is false.

    use_deg2_special : boolean, optional
        If True, use custom, fast implementation specific to degree=2 case.

    prior_var : float, optional
    """
    return _make_approx_fun(lik_fun, data, degree, is_zero, False, timeit,
                            verbose, use_deg2_special, prior_var)


def make_approx_glm_grad_fun(link_grad_fun, data, degree, is_zero=None,
                             timeit=False, verbose=False,
                             use_deg2_special=False):
    return _make_approx_fun(link_grad_fun, data, degree, is_zero, True, timeit,
                            verbose, use_deg2_special, None)


def _make_approx_fun(fun, data, degree, is_zero, grad_mode, timeit, verbose,
                     use_deg2_special, prior_var):
    if timeit is True or timeit == 'all':
        start_time = time.clock()
    elif timeit != 'moments':
        start_time = None
    coeffs = chebyshev_approximation(fun, degree, is_zero)
    if timeit == 'moments':
        start_time = time.clock()
    if degree == 2 and grad_mode == False:
        if verbose:
            print('using Moments2')
        moments = Moments2(data.shape[1], coeffs, sparse=sp.issparse(data))
    else:
        moments = Moments(data.shape[1], degree, coeffs, grad_mode,
                          use_deg2_special)
    moments.add_all(data)
    if verbose:
        if isinstance(moments, Moments):
            print('number of sufficient statistics for degree', degree,
                  'approximation:', moments.moments.size)
            acm = np.abs(moments.coeffs * moments.moments)
            print('num (almost) zeros =', np.sum(acm < 1e-10),
                  'num zeros =', np.sum(acm == 0.0),
                  'max SS =', np.max(acm))
        elif isinstance(moments, Moments2) and sp.issparse(data):
            print('number of nonzeros:',
                  moments.moments[0].shape[0] + moments.moments[1].nnz)
    approx_fun = lambda t: moments.approximate_value_at(t)
    if degree == 2 and prior_var is not None:
        with Timer('mean/cov'):
            meancov = moments.mean_cov(prior_var)
    else:
        meancov = ()
    if start_time is not None:
        total_time = time.clock() - start_time
        return (approx_fun, total_time) + meancov
    else:
        return (approx_fun,) + meancov


# the basis is already normalized
CHEBYSHEV_BASIS = []

def chebyshev_basis(k):
    for i in range(len(CHEBYSHEV_BASIS), k+1):
        coeffs = np.zeros(i+1)
        coeffs[-1] = (1. + np.sign(i)) / np.pi
        CHEBYSHEV_BASIS.append(Chebyshev(coeffs))
    return CHEBYSHEV_BASIS[k]


def chebyshev_bases(k):
    chebyshev_basis(k)
    return CHEBYSHEV_BASIS[:k+1]


def chebyshev_approximation(fun, degree=4, is_zero=None):
    bases = chebyshev_bases(degree)
    approx_coeffs = []
    for i in range(len(bases)):
        if is_zero is None or not is_zero(i):
            approx_coeffs.append(
                quad(lambda x: fun(x) * bases[i](x) / np.sqrt(1 - x**2),
                     -1, 1)[0])
        else:
            approx_coeffs.append(0)
    return Chebyshev(approx_coeffs).convert(kind=Polynomial).coef


### Scalable version ###

def _moment_multiplicities_for(max_deg, partial_mult, mult_sum, degree, exact):
    max_deg = min(max_deg, degree - mult_sum)
    mults = []
    if not exact or sum(partial_mult) == degree:
        mults.append(partial_mult)
    for k in range(max_deg, 0, -1):
        new_mults = _moment_multiplicities_for(k, partial_mult+[k], mult_sum+k,
                                               degree, exact)
        mults.extend(new_mults)
    return mults


def _moment_multiplicities(degree, exact=False):
    # calculate multiplicities for monomials up to degree
    return _moment_multiplicities_for(degree, [], 0, degree, exact)


def multinomial_coeff(ks):
    if len(ks) == 0: return 1
    return (spmisc.factorial(np.sum(ks), exact=True)
            / np.prod(spmisc.factorial(ks, exact=True)))


def num_terms(dim, mults):
    nt = 1
    num_curr_mult = 1
    remaining = dim
    for i in range(len(mults)):
        if i == len(mults) - 1 or mults[i] != mults[i+1]:
            nt *= spmisc.comb(remaining, num_curr_mult, exact=True)
            remaining -= num_curr_mult
            num_curr_mult = 1
        else:
            num_curr_mult += 1
    return nt

def order_two_moments(x):
    dim = x.shape[1]
    moments = np.zeros(1+dim+dim*(dim+1)/2)
    moments[0] = x.shape[0]
    moments[1:dim+1] = np.sum(x**2, axis=0)
    moments[dim+1:2*dim+1] = np.sum(x, axis=0)
    # cross terms
    ind = 2*dim+1
    for i in range(dim-1):
        moments[ind:ind+dim-i-1] = x[:,i].dot(x[:,i+1:])
        ind += dim - i - 1
    return moments


def order_two_moments1d(x):
    dim = x.shape[0]
    moments = np.zeros(1+dim+dim*(dim+1)/2)
    moments[0] = 1
    moments[1:dim+1] = x**2
    moments[dim+1:2*dim+1] = x
    # cross terms
    ind = 2*dim+1
    for i in range(dim-1):
        moments[ind:ind+dim-i-1] = x[i] * x[i+1:]
        ind += dim - i - 1
    return moments


class Moments(object):
    def __init__(self, dim, degree=4, total_multiplicity_coeffs=None,
                 gradient_mode=False, use_deg2_special=False):
        if total_multiplicity_coeffs is not None:
            assert len(total_multiplicity_coeffs) == degree + 1
        else:
            total_multiplicity_coeffs = np.ones(degree + 1)
        self.dim = dim
        self.degree = degree
        self.gradient_mode = gradient_mode
        self.use_deg2_special = use_deg2_special
        self._init_moments(total_multiplicity_coeffs)

    def _init_moments(self, total_multiplicity_coeffs):
        self.multiplicities = _moment_multiplicities(self.degree)
        self.form_dict = dict([(tuple(f),i) for i,f in enumerate(self.multiplicities)])
        self.sizes = [num_terms(self.dim, f) for f in self.multiplicities]
        mhashes = [_ad.mult_hash(array('i', f)) for f in self.multiplicities]
        # make sure hashes are unique
        assert len(mhashes) == len(set(mhashes))
        indices = np.cumsum([0] + self.sizes)[:-1]
        self.mult_indices = dict(zip(mhashes, indices))
        num_moments = spmisc.comb(self.dim + self.degree, self.dim, exact=True)
        if self.gradient_mode:
            moments_shape = (num_moments, self.dim)
        else:
            moments_shape = num_moments
        self.moments = np.zeros(moments_shape)
        coeffs_by_multiplicity = [
            self.sizes[i] * [multinomial_coeff(f)
                             * total_multiplicity_coeffs[sum(f)]]
            for i, f in enumerate(self.multiplicities)]
        self.coeffs = np.array(reduce(lambda a,b: a+b, coeffs_by_multiplicity))

    def mean_cov(self, prior_var):
        if self.degree != 2 or not self.use_deg2_special:
            raise RuntimeError('Can only calculate mean and variance when '
                               'approximation is of degree 2.')
        cmoments = self.coeffs * self.moments
        dim = self.dim
        siginv = np.diagflat(-2 * cmoments[1:dim+1] + 1.0 / prior_var)
        ind = 2*dim + 1
        for i in range(dim-1):
            siginv[i,i+1:] = -cmoments[ind:ind+dim-i-1]
            ind += dim - i - 1
        siginv = (siginv + siginv.T) / 2.0
        sig = np.linalg.inv(siginv)
        musiginv = cmoments[dim+1:2*dim+1]
        # mu = musiginv.dot(sig)
        mu = np.linalg.solve(siginv, musiginv)
        return mu, sig, siginv

    def add(self, x):
        self.moments += self.calculate_moments(x)
        return self

    def add_all(self, xs):
        assert len(xs.shape) == 2
        if self.gradient_mode:
            _ad.sum_moments(None, self.moments, xs, self.mult_indices,
                            self.degree)
        elif self.degree == 2 and self.use_deg2_special:
            self.moments += order_two_moments(xs)
        else:
            _ad.sum_moments(self.moments, None, xs, self.mult_indices,
                            self.degree)
        return self
        # for x in xs:
        #     xm = self.calculate_moments(x)
        #     self.moments += xm

    def calculate_moments(self, x, gradient_mode=None, c_version=True):
        # Flattening will make it possible to do gradient approximations
        # for GLM models much more efficiently since
        # grad_t phi(x . t) = phi'(x . t) x
        if self.degree == 2 and self.use_deg2_special:
            x_moments = order_two_moments1d(x)
        elif not c_version: # XXX: non-c version
            x_moments_lists = [list() for f in self.multiplicities]
            self._build_moments(x_mox_moments_listsents, x)
            x_moments = np.array(reduce(lambda a,b: a+b, x_moments))
        else:  # XXX: c version
            x_moments = np.zeros(self.moments.shape[0])
            _ad.calculate_moments(x_moments, x, self.mult_indices, self.degree)
        if gradient_mode is None:
            gradient_mode = self.gradient_mode
        if gradient_mode:
            return np.outer(x_moments, x)
        else:
            return x_moments


    def _build_moments(self, x_moments, x, multiplicities=[], prev_bar_index=-1,
                       num_bars=0, partial_moment=1):
        # Approach: start with a string of dim + degree o's. Then turn dim o's
        # into |'s. The number of o's between the i and i+1 |'s is the
        # multiplicity of x[i].

        # base case: have put down all the bars
        if num_bars == x.shape[0]:
            multi = self.degree + x.size - prev_bar_index - 1
            if multi > 0:
                moment = partial_moment * x[-1] ** multi
                multiplicities.append(multi)
            else:
                moment = partial_moment
            # print((1+num_bars) * " ", num_bars, prev_bar_index, self.degree + x.size, multiplicities, moment)
            mults = list(multiplicities)
            mults.sort(reverse=True)
            x_moments[self.form_dict[tuple(mults)]].append(moment)
            if multi > 0:
                multiplicities.pop()
        else:
            num_bars += 1
            for i in range(prev_bar_index + 1, self.degree + num_bars):
                multi = i - prev_bar_index - 1
                if multi > 0 and num_bars > 1:
                    updated_partial_moment = partial_moment * x[num_bars-2] ** multi
                    multiplicities.append(multi)
                else:
                    updated_partial_moment = partial_moment
                # print(num_bars * " ", num_bars, prev_bar_index, i, multiplicities)
                self._build_moments(x_moments, x, multiplicities, i,
                                    num_bars, updated_partial_moment)
                if multi > 0 and num_bars > 1:
                    multiplicities.pop()

    def approximate_value_at(self, y):
        assert y.ndim == 1, "y must be 1-dimensional"
        assert y.shape[0] == self.dim, "y has incorrect dimension"
        y_moments = self.calculate_moments(y, False)
        return (self.coeffs * y_moments).dot(self.moments)
        # value = 0
        # for mult, c, ym, m in zip(moments.multiplicities, moments.coeffs, y_moments, moments.moments):
        #     index = np.sum(mult)
        #     if coeffs[index] != 0:
        #         value += coeffs[index] * c * np.sum(ym * m)
        # return value

class MomentList(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, args)
        self.sparse = kwargs.get('sparse', False)

    def sum(self):
        return reduce(lambda a,b: a + np.sum(b), self, 0.0)

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return MomentList(*[a + other for a in self], sparse=self.sparse)
        elif isinstance(other, MomentList):
            if len(self) != len(other):
                raise ValueError("MomentLists aren't the same length")
            return MomentList(*[a+b for a,b in zip(self, other)],
                              sparse=self.sparse)
        else:
            raise ValueError('Invalid type to add')

    def __radd__(self, other):
        return self.__add_(other)

    def __iadd__(self, other):
        for i in range(len(self)):
            self[i] += other[i]
        return self

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return MomentList(*[a * other for a in self], sparse=self.sparse)
        elif isinstance(other, MomentList):
            if len(self) != len(other):
                raise ValueError("MomentLists aren't the same length")
            if self.sparse:
                return MomentList(*[a.multiply(b) for a,b in zip(self, other)],
                                  sparse=self.sparse)
            else:
                return MomentList(*[a*b for a,b in zip(self, other)],
                                  sparse=self.sparse)
        else:
            raise ValueError('Invalid type to multiply')

    def __rmul__(self, other):
        return self.__mul_(other)

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            other = [other] * len(self)
        for i in range(len(self)):
            if self.sparse:
                self[i] = self[i].multiply(other[i])
            else:
                self[i] *= other[i]
        return self


# specialized Moments object for degree=2 case only
class Moments2(object):
    def __init__(self, dim, total_multiplicity_coeffs=None, sparse=False):
        if total_multiplicity_coeffs is not None:
            assert len(total_multiplicity_coeffs) == 3
        else:
            total_multiplicity_coeffs = np.ones(3)
        self.dim = dim
        self.sparse = sparse
        if sparse:
            self.moments = MomentList(sp.csc_matrix((dim,1)),
                                      sp.csc_matrix((dim, dim)),
                                      sparse=sparse)
        else:
            self.moments = MomentList(np.zeros(dim), np.zeros((dim, dim)),
                                      sparse=sparse)
        self.coeffs = dict()
        self._init_coeffs(total_multiplicity_coeffs)

    def _init_coeffs(self, total_multiplicity_coeffs, index=0):
        if self.sparse:
            self.coeffs[index] = total_multiplicity_coeffs
        else:
            mean_coeffs = np.full(self.dim, total_multiplicity_coeffs[1])
            cov_coeffs = np.full((self.dim, self.dim), total_multiplicity_coeffs[2])
            self.coeffs[index] = MomentList(mean_coeffs, cov_coeffs,
                                            sparse=self.sparse)

    def add(self, x, index=0):
        self.moments += self.coeffs[index] * self.calculate_moments(x)
        return self

    def add_all(self, xs, index=0):
        assert len(xs.shape) == 2
        # if self.sparse:
        #     xs_sum = xs.sum(axis=0).reshape((-1,1))
        #     xs_outer = xs.transpose() * xs
        #     x_moments = MomentList(xs_sum, xs_outer,
        #                            sparse=self.sparse)
        #     x_moments[0] *= self.coeffs[index][1]
        #     x_moments[1] *= self.coeffs[index][2]
        # else:
        #     x_moments = MomentList(np.sum(xs, axis=0), np.dot(xs.T, xs),
        #                            sparse=self.sparse)
        #     x_moments *= self.coeffs[index]
        x_moments = calculate_x_moments_local(self.coeffs[index], xs, self.sparse)
        self.moments += x_moments
        return self

    def calculate_moments(self, x, gradient_mode=None):
        if self.sparse:
            return MomentList(x.copy(), x.transpose() * x, sparse=self.sparse)
        else:
            return MomentList(x.copy(), np.outer(x, x), sparse=self.sparse)

    def approximate_value_at(self, y):
        assert y.ndim == 1 or y.shape[1] == 1, "y must be 1-dimensional"
        assert y.shape[0] == self.dim, "y has incorrect dimension"
        # TODO: in sparse case, only calculate the cross terms that we need
        # (probably want to do this with cython)
        y_moments = self.calculate_moments(y, None)
        y_moments *= self.moments
        return y_moments.sum()

    def mean_cov(self, prior_var):
        dim = self.dim
        m2 = self.moments[1]
        if self.sparse:
            siginv = -m2 - sp.diags(m2.diagonal()) + sp.eye(dim) / prior_var
            # TODO: efficient implementation of this
            sig = np.linalg.inv(siginv.toarray())
        else:
            siginv = -m2 - np.diag(np.diag(m2)) + np.eye(dim) / prior_var
            sig = np.linalg.inv(siginv)
        musiginv = self.moments[0]
        if self.sparse:
            mu = sp.linalg.spsolve(siginv, np.array(musiginv))
        else:
            mu = np.linalg.solve(siginv, musiginv)
        return mu, sig, siginv


@ray.remote
def calculate_x_moments(coeffs, xs, sparse, ys=None):
    return calculate_x_moments_local(coeffs, xs, sparse, ys)


def calculate_x_moments_local(coeffs, xs, sparse, ys=None):
    if ys is not None:
        if sp.issparse(xs):
            xs = sp.diags(ys).dot(xs)
        else:
            xs = ys[:, np.newaxis] * xs
    if sparse:
        xs_sum = xs.sum(axis=0).reshape((-1,1))
        xs_outer = xs.transpose() * xs
        x_moments = MomentList(xs_sum, xs_outer,
                               sparse=sparse)
        x_moments[0] *= coeffs[1]
        x_moments[1] *= coeffs[2]
    else:
        x_moments = MomentList(np.sum(xs, axis=0), np.dot(xs.T, xs),
                               sparse=sparse)
        x_moments *= coeffs
    return x_moments

#
# # specialized Moments object for degree=2 case only
# class Moments2(object):
#     def __init__(self, dim, total_multiplicity_coeffs=None, sparse=False,
#                  distributed=False):
#         if total_multiplicity_coeffs is not None:
#             assert len(total_multiplicity_coeffs) == 3
#         else:
#             total_multiplicity_coeffs = np.ones(3)
#         self.dim = dim
#         self.sparse = sparse
#         if sparse:
#             order_one = sp.csc_matrix((dim,1))
#             order_two = sp.csc_matrix((dim, dim))
#         else:
#             order_one = np.zeros(dim)
#             order_two = np.zeros((dim, dim))
#         if distributed:
#             self.moments = DistributedMomentList(sparse, order_one, order_two)
#         else:
#             self.moments = MomentList(order_one, order_two, sparse=sparse)
#         self.distributed = distributed
#         self.coeffs = dict()
#         self.datapoints_seen = 0
#         self._init_coeffs(total_multiplicity_coeffs)
#
#     def _init_coeffs(self, total_multiplicity_coeffs, index=0):
#         if self.sparse:
#             self.coeffs[index] = total_multiplicity_coeffs
#         else:
#             mean_coeffs = np.full(self.dim, total_multiplicity_coeffs[1])
#             cov_coeffs = np.full((self.dim, self.dim), total_multiplicity_coeffs[2])
#             self.coeffs[index] = MomentList(mean_coeffs, cov_coeffs,
#                                             sparse=self.sparse)
#
#     def add(self, x, index=0):
#         self.moments += self.coeffs[index] * self.calculate_moments(x)
#         self.datapoints_seen += 1
#         self.moments.datapoints_added(1)
#         return self
#
#     def add_all(self, xs, index=0):
#         assert len(xs.shape) == 2
#         self.datapoints_seen += xs.shape[0]
#         if self.distributed:
#             return distributed_add_all.remote(self.moments, self.coeffs[index],
#                                               xs, self.sparse)
#         else:
#             x_moments = calculate_x_moments_local(self.coeffs[index], xs, self.sparse)
#             self.moments += x_moments
#             self.moments.datapoints_added(xs.shape[0])
#             return self
#
#     def calculate_moments(self, x, gradient_mode=None):
#         if self.sparse:
#             return MomentList(x.copy(), x.transpose() * x, sparse=self.sparse)
#         else:
#             return MomentList(x.copy(), np.outer(x, x), sparse=self.sparse)
#
#     def approximate_value_at(self, y):
#         assert y.ndim == 1 or y.shape[1] == 1, "y must be 1-dimensional"
#         assert y.shape[0] == self.dim, "y has incorrect dimension"
#         # TODO: in sparse case, only calculate the cross terms that we need
#         # (probably want to do this with cython)
#         y_moments = self.calculate_moments(y, None)
#         y_moments *= self.moments
#         return y_moments.sum()
#
#     def up_to_date(self):
#         return self.datapoints_seen == self.moments.datapoints_seen
#
#     def mean_cov(self, prior_var):
#         dim = self.dim
#         m2 = self.moments[1]
#         if self.sparse:
#             siginv = -m2 - sp.diags(m2.diagonal()) + sp.eye(dim) / prior_var
#             # TODO: efficient implementation of this
#             sig = np.linalg.inv(siginv.toarray())
#         else:
#             siginv = -m2 - np.diag(np.diag(m2)) + np.eye(dim) / prior_var
#             sig = np.linalg.inv(siginv)
#         musiginv = self.moments[0]
#         if self.sparse:
#             mu = sp.linalg.spsolve(siginv, np.array(musiginv))
#         else:
#             mu = np.linalg.solve(siginv, musiginv)
#         return mu, sig, siginv


def test_grad_moments(dim=3, N=10000, degree=4, max_value=1, num_thetas=100):
    phi = lambda x: 1.0 / (1.0 + np.exp(max_value * x))

    data = np.random.rand(N, dim) / np.sqrt(dim)
    thetas = np.random.rand(num_thetas, dim) / np.sqrt(dim)

    with Timer('coefficient calculation') as t_cc:
        coeffs = chebyshev_approximation(phi, degree,
                                         lambda x: x > 1 and x % 2 == 0)
    with Timer('moments calculation    ') as t_mc:
        moments = Moments(dim, degree, coeffs, True)
        moments.add_all(data)
    with Timer('approximate gradient   ') as t_al:
        approx_grads = np.array(
            [moments.approximate_value_at(thetas[i,:])
             for i in range(num_thetas)])
    with Timer('exact gradient         ') as t_el:
        grads = np.array(
            [logistic_likelihood_grad(max_value*thetas[i,:], data)
             for i in range(num_thetas)])

    avg_err = np.mean(np.linalg.norm(approx_grads - grads, axis=1))
    time_diff = (t_el.interval - t_al.interval) / num_thetas
    if time_diff <= 0:
        breakeven = None
    else:
        breakeven = int(t_mc.interval / time_diff + 1)

    print()
    if breakeven is None:
        print('no breakeven possible')
    else:
        print('breakeven after %d likelihood evals' % breakeven)
    # print()
    # print('approximate likelihoods:', approx_liks)
    # print('exact likelihoods:      ', liks)
    print('average relative error: ', avg_err)


def test_moments(dim=3, N=10000, degree=4, max_value=1, num_thetas=10):
    phi = lambda x: -np.log1p(np.exp(-max_value * x))

    np.random.seed(919)
    data = np.random.rand(N, dim) / np.sqrt(dim)
    thetas = np.random.rand(num_thetas, dim) / np.sqrt(dim)

    with Timer('coefficient calculation') as t_cc:
        coeffs = chebyshev_approximation(phi, degree,
                                         lambda x: x > 1 and x % 2 == 1)
    with Timer('moments calculation    ') as t_mc:
        moments = Moments(dim, degree, coeffs)
        moments.add_all(data)
    with Timer('approximate likelihood ') as t_al:
        approx_liks = [moments.approximate_value_at(thetas[i,:])
                       for i in range(num_thetas)]
    with Timer('exact likelihood       ') as t_el:
        liks = [logistic_likelihood(max_value * thetas[i,:], data)
                for i in range(num_thetas)]

    avg_err = np.mean(np.abs((np.array(liks) - approx_liks) / liks))
    time_diff = (t_el.interval - t_al.interval) / num_thetas
    if time_diff <= 0:
        breakeven = None
    else:
        breakeven = int(t_mc.interval / time_diff + 1)

    print()
    if breakeven is None:
        print('no breakeven possible')
    else:
        print('breakeven after %d likelihood evals' % breakeven)
    print()
    print('approximate likelihoods:', approx_liks)
    print('exact likelihoods:      ', liks)
    print('average relative error: ', avg_err)


# XXX work in progress
def test_pass_distributed(dim=3, N=1000, chunks=10, num_cpus=2, max_value=1, num_thetas=10):
    phi = lambda x: -np.log1p(np.exp(-max_value * x))

    np.random.seed(919)
    Xs = [np.random.randn(N, dim) / np.sqrt(dim) for i in range(chunks)]
    ys = [np.random.choice([-1,1], N) for i in range(chunks)]

    thetas = np.random.rand(num_thetas, dim) / np.sqrt(dim)

    coeffs = chebyshev_approximation(phi, 2)
    offset = N * coeffs[0]
    moments2 = Moments2(dim, coeffs)
    # XXX: still need to update code from here to end
    for i in range(chunks):
        moments2.add_all()


    with Timer('approximate likelihood ') as t_al:
        approx_liks = [moments.approximate_value_at(thetas[i,:])
                       for i in range(num_thetas)]
    with Timer('approximate2 likelihood') as t_al2:
        approx2_liks = [offset + moments2.approximate_value_at(thetas[i,:])
                       for i in range(num_thetas)]
    with Timer('exact likelihood       ') as t_el:
        liks = [logistic_likelihood(max_value * thetas[i,:], data)
                for i in range(num_thetas)]

    avg_err = np.mean(np.abs((np.array(liks) - approx_liks) / liks))
    time_diff = (t_el.interval - t_al.interval) / num_thetas
    if time_diff <= 0:
        breakeven = None
    else:
        breakeven = int(t_mc.interval / time_diff + 1)
    time_diff2 = (t_el.interval - t_al2.interval) / num_thetas
    if time_diff2 <= 0:
        breakeven2 = None
    else:
        breakeven2 = int(t_mc2.interval / time_diff2 + 1)

    print()
    if breakeven is None:
        print('no breakeven possible using Moments')
    else:
        print('breakeven for Moments after %d likelihood evals' % breakeven)
    if breakeven2 is None:
        print('no breakeven possible using Moments2')
    else:
        print('breakeven for Moments2 after %d likelihood evals' % breakeven2)
    print()
    print('approximate likelihoods: ', approx_liks)
    print('approximate2 likelihoods:', approx2_liks)
    print('exact likelihoods:       ', liks)
    print('average relative error:  ', avg_err)


def test_moments2(dim=3, N=1000, max_value=1, num_thetas=10):
    phi = lambda x: -np.log1p(np.exp(-max_value * x))

    np.random.seed(919)
    data = np.random.rand(N, dim) / np.sqrt(dim)
    thetas = np.random.rand(num_thetas, dim) / np.sqrt(dim)

    with Timer('coefficient calculation') as t_cc:
        coeffs = chebyshev_approximation(phi, 2)
        offset = N * coeffs[0]
    with Timer('moments calculation    ') as t_mc:
        moments = Moments(dim, 2, coeffs, use_deg2_special=True)
        moments.add_all(data)
    with Timer('moments2 calculation   ') as t_mc2:
        moments2 = Moments2(dim, coeffs)
        moments2.add_all(data)
    with Timer('approximate likelihood ') as t_al:
        approx_liks = [moments.approximate_value_at(thetas[i,:])
                       for i in range(num_thetas)]
    with Timer('approximate2 likelihood') as t_al2:
        approx2_liks = [offset + moments2.approximate_value_at(thetas[i,:])
                       for i in range(num_thetas)]
    with Timer('exact likelihood       ') as t_el:
        liks = [logistic_likelihood(max_value * thetas[i,:], data)
                for i in range(num_thetas)]

    avg_err = np.mean(np.abs((np.array(liks) - approx_liks) / liks))
    time_diff = (t_el.interval - t_al.interval) / num_thetas
    if time_diff <= 0:
        breakeven = None
    else:
        breakeven = int(t_mc.interval / time_diff + 1)
    time_diff2 = (t_el.interval - t_al2.interval) / num_thetas
    if time_diff2 <= 0:
        breakeven2 = None
    else:
        breakeven2 = int(t_mc2.interval / time_diff2 + 1)

    print()
    if breakeven is None:
        print('no breakeven possible using Moments')
    else:
        print('breakeven for Moments after %d likelihood evals' % breakeven)
    if breakeven2 is None:
        print('no breakeven possible using Moments2')
    else:
        print('breakeven for Moments2 after %d likelihood evals' % breakeven2)
    print()
    print('approximate likelihoods: ', approx_liks)
    print('approximate2 likelihoods:', approx2_liks)
    print('exact likelihoods:       ', liks)
    print('average relative error:  ', avg_err)

    # mu, cov = moments.mean_cov(1.0)
    # mu2, cov2 = moments2.mean_cov(1.0)
    # print()
    # print('mean :', mu)
    # print('mean2:', mu2)
    # print('var  :', np.diag(cov))
    # print('var2 :', np.diag(cov2))


def check_mode_accuracy(dim=3, N=1000, degree=4, max_value=1):
    assert dim <= 10
    np.random.seed(1)
    phi = lambda x: np.log1p(np.exp(-max_value * x))
    theta = np.array([-3, 1.2, -.5, .8, -1.,-.7,  3.,   4.,  3.5,  4.5])
    probs = np.array([ 1,  .2,  .3, .5, .1,  .2, .01, .007, .005, .001])
    X, y = generate_binary_data(N, probs[:dim], theta[:dim])
    if sp.issparse(X):
        data = sp.diags(y).dot(X).toarray()
    else:
        data = y[:, np.newaxis] * X
    data /= np.sqrt(dim)

    def prior(t):
        tnorm = np.linalg.norm(t)
        if tnorm > 1:
            return np.inf
        else:
            return max_value * tnorm**2
    t0 = np.zeros(dim)

    with Timer('approximate optimization'):
        coeffs = chebyshev_approximation(phi, degree,
                                         lambda x: x > 1 and x % 2 == 1)
        moments = Moments(dim, degree, coeffs)
        moments.add_all(data)
        approx_fun = lambda t: moments.approximate_value_at(t) + prior(t)
        approx_res = minimize(approx_fun, t0, method='BFGS')

    with Timer('exact optimization'):
        true_fun = lambda t: -logistic_likelihood(max_value * t, data) + prior(t)
        true_res = minimize(true_fun, t0, method='BFGS')

    scale_factor = max_value / np.sqrt(dim)
    print('true mode:  ', scale_factor * true_res.x, ', norm =',
          max_value * np.linalg.norm(true_res.x))
    print('  error:', np.linalg.norm(scale_factor * true_res.x - theta[:dim]))
    print('approx mode:', scale_factor * approx_res.x, ', norm =',
          max_value * np.linalg.norm(approx_res.x))
    print('  error:', np.linalg.norm(scale_factor * approx_res.x - theta[:dim]))
    print('true mode value:   ', true_res.fun)
    print('approx mode value: ', true_fun(approx_res.x))


### Basic version ###
# very inefficient except when d and k are both small

def calculate_moments(x, degree=4):
    if degree == 0:
        return [np.ones(1)]
    moments = [np.ones(1), x.copy()]
    for i in range(degree - 1):
        moments.append(np.tensordot(x, moments[-1], axes=0))
    return moments


def update_moments(moments, xs):
    if xs.ndim == 1:
        _update_moments(moments, xs)
    else:
        for i in range(xs.shape[0]):
            _update_moments(moments, xs[i,:])


def _update_moments(moments, x):
    xm = calculate_moments(x, degree=len(moments)+1)
    for i in range(len(moments)):
        moments[i] += xm[i]


def approximate_value(coeffs, moments, y):
    degree = len(moments) - 1
    assert len(coeffs) == degree + 1
    ym = calculate_moments(y, degree)
    value = 0
    for i in range(degree + 1):
        if coeffs[i] != 0:
            value += coeffs[i] * np.sum(ym[i] * moments[i])
    return value
