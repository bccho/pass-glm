from __future__ import absolute_import
from warnings import warn

import numpy as np
import numpy.random as npr
from numpy.testing import *
from scipy.stats import norm

import passglm.inference as inf

npr.seed(9)

def _estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    print 'result[0] =', result[0]
    inds = np.where(result < 0)[0]
    if len(inds) > 0:
        return np.sum(result[:inds[0]])
    else:
        return np.sum(result)


def test_simple_mh_geweke():
    mu = 0.0
    sig = 2.0
    p = norm(mu, sig).pdf

    f = lambda x: -0.5 * x**2 / sig**2

    # Define a spherical proposal function
    q = None
    sample_q = lambda x: x + 1. * np.random.randn()

    steps = 100000
    warmup = 1000
    thin = 3
    samples, accept_rate = np.array(inf.mh(0, f, None, sample_q,
                                           steps, warmup, thin))
    samples = np.array(samples)
    ac = _estimated_autocorrelation(samples)
    atol = 2 * sig * ac / np.sqrt(len(samples))

    expected_num_samples = (steps - warmup + thin - 1) / thin
    assert_array_equal(expected_num_samples, len(samples))
    assert_allclose(mu, np.mean(samples), atol=atol)
    assert_allclose(sig, np.std(samples), atol=atol)
    assert accept_rate > 0.0 and accept_rate < 1.0


@dec.slow
def test_simple_mh_2d_geweke():
    dim = 2
    mu = np.array([-1., 2.])
    sig = 2
    f = lambda x: -0.5 * np.sum((x - mu)**2) / sig**2

    # Define a spherical proposal function
    q = None
    sample_q = lambda x: x + 1.*np.random.randn(dim)

    steps = 100000
    warmup = 1000
    thin = 2
    x0 = np.zeros(dim)
    samples, accept_rate = np.array(inf.mh(x0, f, None, sample_q,
                                           steps, warmup, thin))
    samples = np.array(samples)
    ac = _estimated_autocorrelation(samples[:,0])
    atol = 2 * sig * ac / np.sqrt(samples.shape[0])

    expected_num_samples = (steps - warmup + thin - 1) / thin
    assert_array_equal(expected_num_samples, samples.shape[0])
    assert_allclose(mu, np.mean(samples, 0), atol=atol)
    assert_allclose(sig, np.std(samples, 0), atol=atol)
    assert accept_rate > 0.0 and accept_rate < 1.0


@dec.slow
def test_adaptive_mh_2d_geweke():
    dim = 2
    mu = np.array([-1., 2.])
    sig = 2
    f = lambda x: -0.5 * np.sum((x - mu)**2) / sig**2

    # Define a spherical proposal function
    q = None
    sample_q = lambda x, ell: x + np.exp(ell)*np.random.randn(dim)

    steps = 100000
    x0 = np.zeros(dim)
    ell0 = 0.0
    target_rate = 0.234
    samples, accept_rate = np.array(inf.mh(x0, f, None, sample_q, steps,
                                           proposal_param=ell0,
                                           target_rate=target_rate))
    samples = np.array(samples)
    ac = _estimated_autocorrelation(samples[:,0])
    atol = 2 * sig * ac / np.sqrt(samples.shape[0])

    expected_num_samples = (steps + 1) / 2
    assert_array_equal(expected_num_samples, samples.shape[0])
    assert_allclose(mu, np.mean(samples, 0), atol=atol)
    assert_allclose(sig, np.std(samples, 0), atol=atol)
    assert_allclose(target_rate, accept_rate, atol=.05)
