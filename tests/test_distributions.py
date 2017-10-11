from __future__ import absolute_import

import numpy as np
import numpy.random as npr
from numpy.testing import *

import passglm.distributions as dists

npr.seed(1)

def logistic_likelihood_test():
    Z = np.array([[0., 1., -1.],
                  [10., -2., 1.]])
    weights = np.array([3., .5])
    theta = np.array([2., -.5, 1.5])

    ip0 = np.sum(Z[0,:] * theta)
    ip1 = np.sum(Z[1,:] * theta)
    true_liks = -np.log(1 + np.exp(-np.array([ip0, ip1])))

    true_lik = np.sum(true_liks)
    true_wlik = np.sum(weights * true_liks)

    lik = dists.logistic_likelihood(theta, Z)
    wlik = dists.logistic_likelihood(theta, Z, weights)

    assert_allclose(true_lik, lik)
    assert_allclose(true_wlik, wlik)
