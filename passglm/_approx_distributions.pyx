# distutils: language=c++
# cython: boundscheck=True
# cython: wraparound=False
# cython: cdivision=True
# cython: linetrace=False
# cython: binding=False
from __future__ import print_function
from cpython cimport array
import array
from libcpp.map cimport map
import numpy as np
cimport numpy as np
cimport cython
#from cpython cimport bool

# from .profiling import line_profiled
# PROFILING = False

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

np.import_array()

cdef INT MAX_VALUE = np.iinfo(np.int32).max
cdef INT LARGE_PRIME = 49979687
cdef INT LARGE_NUMBER = 39145639

cpdef inline INT mult_hash(array.array a):
  #return sum([101**x for x in a]) % MAX_VALUE
  # slightly faster version
  cdef:
    long h = 0
  for x in a:
    h += pow(LARGE_NUMBER, x, LARGE_PRIME)
    h %= LARGE_PRIME
  return h


# def sum_moments(np.ndarray[DOUBLE, ndim=1] x_moments,
#                 np.ndarray[DOUBLE, ndim=2] xs,
#                 map[INT, INT] mult_indices,
#                 INT degree,
#                 bool gradient_mode):
#     cdef:
#       np.ndarray[DOUBLE, ndim=1] x
#     for x in xs:
#       calculate_moments(x_moments, x, mult_indices, degree, gradient_mode)

def sum_moments(np.ndarray[DOUBLE, ndim=1] x_moments,
                np.ndarray[DOUBLE, ndim=2] x_grad_moments,
                np.ndarray[DOUBLE, ndim=2] x,
                map[INT, INT] mult_indices,
                INT degree):
    cdef:
      array.array mults = array.array('i')
      np.ndarray[DOUBLE, ndim=1] partial_moments = np.ones(x.shape[0])
    sum_moments_rec(x_moments, x_grad_moments, x, x.shape[1], mult_indices,
                    degree, mults, -1, 0, partial_moments)


cdef void sum_moments_rec(np.ndarray[DOUBLE, ndim=1] x_moments,
                          np.ndarray[DOUBLE, ndim=2] x_grad_moments,
                          np.ndarray[DOUBLE, ndim=2] x,
                          INT dim,
                          map[INT, INT]& mult_indices,
                          INT degree,
                          array.array multiplicities,
                          INT prev_bar_index,
                          INT num_bars,
                          np.ndarray[DOUBLE, ndim=1] partial_moments) except *:
    # Approach: start with a string of dim + degree o's. Then turn dim o's
    # into |'s. The number of o's between the i and i+1 |'s is the
    # multiplicity of x[i].
    cdef:
      int mhash_val
      np.ndarray[DOUBLE, ndim=1] updated_partial_moments

    num_bars += 1
    for i in range(prev_bar_index + 1, degree + num_bars):
        multi = i - prev_bar_index - 1
        if multi > 0 and num_bars > 1:
            updated_partial_moments = partial_moments * x[:,num_bars-2] ** multi
            multiplicities.append(multi)
        else:
            updated_partial_moments = partial_moments
        # print(num_bars * " ", num_bars, prev_bar_index, i, multiplicities, updated_partial_moment)
        # XXX: if we want to cut off when partial moment is zero, need to
        # increment mult_indices, which seems hard to do if
        # updated_partial_moment != 0:
        if num_bars == dim:  # base case: have put down all the bars
            final_multi = degree + dim - i - 1
            if final_multi > 0:
                moments = updated_partial_moments * x[:,dim - 1] ** final_multi
                multiplicities.append(final_multi)
            else:
                moments = updated_partial_moments
            # print((1+num_bars) * " ", num_bars, prev_bar_index, degree + dim, multiplicities, moment)
            mhash_val = mult_hash(multiplicities)
            if x_moments is not None:
              x_moments[mult_indices[mhash_val]] += np.sum(moments)
              # print(multiplicities, x_moments[mult_indices[mhash_val]])
            if x_grad_moments is not None:
              x_grad_moments[mult_indices[mhash_val],:] = moments.dot(x)
            mult_indices[mhash_val] += 1
            if final_multi > 0:
                multiplicities.pop()
        else:
            sum_moments_rec(x_moments, x_grad_moments, x, dim, mult_indices,
                            degree, multiplicities, i, num_bars,
                            updated_partial_moments)
        if multi > 0 and num_bars > 1:
            multiplicities.pop()


def calculate_moments(np.ndarray[DOUBLE, ndim=1] x_moments,
                      np.ndarray[DOUBLE, ndim=1] x,
                      map[INT, INT] mult_indices,
                      INT degree):
    cdef:
      array.array mults = array.array('i')
    build_moments_rec(x_moments, x, x.size, mult_indices, degree, mults,
                      -1, 0, 1.0)


# @line_profiled
# def build_moments_rec(np.ndarray[DOUBLE, ndim=1] x_moments,
#                       np.ndarray[DOUBLE, ndim=1] x,
#                       INT dim,
#                       map[INT, INT] mult_indices,
#                       INT degree,
#                       array.array multiplicities,
#                       INT prev_bar_index,
#                       INT num_bars,
#                       DOUBLE partial_moment):
cdef void build_moments_rec(np.ndarray[DOUBLE, ndim=1] x_moments,
                            np.ndarray[DOUBLE, ndim=1] x,
                            INT dim,
                            map[INT, INT]& mult_indices,
                            INT degree,
                            array.array multiplicities,
                            INT prev_bar_index,
                            INT num_bars,
                            DOUBLE partial_moment) except *:
    # Approach: start with a string of dim + degree o's. Then turn dim o's
    # into |'s. The number of o's between the i and i+1 |'s is the
    # multiplicity of x[i].
    cdef:
      int mhash_val
      DOUBLE updated_partial_moment

    num_bars += 1
    for i in range(prev_bar_index + 1, degree + num_bars):
        multi = i - prev_bar_index - 1
        if multi > 0 and num_bars > 1:
            updated_partial_moment = partial_moment * x[num_bars-2] ** multi
            multiplicities.append(multi)
        else:
            updated_partial_moment = partial_moment
        # print(num_bars * " ", num_bars, prev_bar_index, i, multiplicities, updated_partial_moment)
        # XXX: if we want to cut off when partial moment is zero, need to
        # increment mult_indices, which seems hard to do if
        # updated_partial_moment != 0:
        if num_bars == dim:  # base case: have put down all the bars
            final_multi = degree + dim - i - 1
            if final_multi > 0:
                moment = updated_partial_moment * x[dim - 1] ** final_multi
                multiplicities.append(final_multi)
            else:
                moment = updated_partial_moment
            # print((1+num_bars) * " ", num_bars, prev_bar_index, degree + dim, multiplicities, moment)
            mhash_val = mult_hash(multiplicities)
            x_moments[mult_indices[mhash_val]] += moment
            mult_indices[mhash_val] += 1
            if final_multi > 0:
                multiplicities.pop()
        else:
            build_moments_rec(x_moments, x, dim, mult_indices, degree,
                              multiplicities, i, num_bars,
                              updated_partial_moment)
        if multi > 0 and num_bars > 1:
            multiplicities.pop()
