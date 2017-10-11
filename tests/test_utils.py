import os

import numpy.random as npr
from numpy.testing import *
import scipy.sparse as sp

from passglm.utils import *


def test_create_folder_if_not_exist():
    try:
        print 'Creating test folder...'
        create_folder_if_not_exist('test_folder')
        assert os.path.exists('test_folder')
        print 'Trying to recreate test folder...'
        create_folder_if_not_exist('test_folder')
        assert os.path.exists('test_folder')
    finally:
        os.removedirs('test_folder')


def test_call_with_superset_args():
    def test_fun(a, b):
        return a+b
    args = {'a' : 4, 'b' : 2, 'c' : 'a', 'x' : 3}
    assert 6 == call_with_superset_args(test_fun, args)


def test_ensure_dimension_matches():
    A = npr.rand(20, 10)
    B = npr.rand(30, 5)
    AA, BB = ensure_dimension_matches(A, B, axis=0)
    assert AA.shape[0] == BB.shape[0]
    assert_allclose(B, BB)
    assert_allclose(A, AA[:A.shape[0],:])
    assert_allclose(0, AA[A.shape[0]:,:])

    AA, BB = ensure_dimension_matches(A, B, axis=1)
    assert AA.shape[1] == BB.shape[1]
    assert_allclose(A, AA)
    assert_allclose(B, BB[:,:B.shape[1]])
    assert_allclose(0, BB[:,B.shape[1]:])

    BB, AA = ensure_dimension_matches(B, A, axis=0)
    assert AA.shape[0] == BB.shape[0]
    assert_allclose(B, BB)
    assert_allclose(A, AA[:A.shape[0],:])
    assert_allclose(0, AA[A.shape[0]:,:])

    BB, AA = ensure_dimension_matches(B, A, axis=1)
    assert AA.shape[1] == BB.shape[1]
    assert_allclose(A, AA)
    assert_allclose(B, BB[:,:B.shape[1]])
    assert_allclose(0, BB[:,B.shape[1]:])

    A = sp.rand(20, 10)
    B = sp.rand(30, 5)
    BB, AA = ensure_dimension_matches(B, A, axis=1)
    A, B = A.toarray(), B.toarray()
    AA, BB = AA.toarray(), BB.toarray()
    assert AA.shape[1] == BB.shape[1]
    assert_allclose(A, AA)
    assert_allclose(B, BB[:,:B.shape[1]])
    assert_allclose(0, BB[:,B.shape[1]:])
