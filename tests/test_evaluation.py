from __future__ import absolute_import
from warnings import warn

import numpy as np
import numpy.random as npr
from numpy.testing import *

import passglm.evaluation as ceval


npr.seed(1)


def test_mean_relative_squared_error():
    t_stdev_list = [0.5, 2.0]
    t_mean_list = [-1.0, 2.0]
    o_mean_list = [0.0, 6.0]

    for t_stdev in t_stdev_list:
        for t_mean in t_mean_list:
            for o_mean in o_mean_list:
                target = t_mean + t_stdev * npr.randn(10000, 10)
                other = o_mean + npr.randn(10000, 10)
                expected_mrse = (t_mean - o_mean)**2 / t_stdev**2
                mrse = ceval.mean_relative_squared_error(target, other,
                                                         None, None)
                print t_stdev, t_mean, o_mean
                print np.mean(target, 0)
                print np.mean(other, 0)
                print np.var(target, 0)
                assert_allclose(expected_mrse, mrse, rtol=.05)
