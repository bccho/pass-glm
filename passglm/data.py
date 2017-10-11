# Authors: Jonathan Huggins <jhuggins@mit.edu>
#          Trevor Campbell <tdjc@mit.edu>

from __future__ import absolute_import, print_function

import sys
import csv
import hashlib
import cPickle as cpk
from warnings import warn

import numpy as np
import numpy.random as npr
import scipy.sparse as sp

import sklearn.datasets as skl_ds
from sklearn import preprocessing
from .distributions import logistic_likelihood
from .utils import ensure_dimension_matches

import h5py

# based on: http://stackoverflow.com/questions/8955448/
def save_sparse_Xy(filename, X, y):
    """Save sparse X and array-like y as an npz file.

    Parameters
    ----------
    filename : string

    X : sparse matrix, shape=(n_samples, n_features)

    y : array-like, shape=(n_samples,)
    """
    np.savez(filename, data=X.data, indices=X.indices, indptr=X.indptr,
             shape=X.shape, y=y)


def save_Xy(filename, X, y):
    """Save X, y as an npz file.

    Parameters
    ----------
    filename : string

    X : matrix-like, shape=(n_samples, n_features)

    y : array-like, shape=(n_samples,)
    """
    if sp.issparse(X):
        save_sparse_Xy(filename, X, y)
    else:
        np.savez(filename, X=X, y=y)


def _load_svmlight_data(path):
    X, y = skl_ds.load_svmlight_file(path)
    return X, y


def _load_npy_data(path):
    xy = np.load(path)
    X = xy[:, :-1]
    y = xy[:, -1]
    return X, y


def _load_npz_data(path):
    loader = np.load(path)
    if 'X' in loader:
        X = loader['X']
    else:
        X = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])
    y = loader['y']
    return X, y


def _load_hdf5_data(path):
    f = h5py.File(path, 'r')
    X = f['x']
    y = f['y']
    f.close()
    return X, y


def _load_csv_data(path):
    xy = np.genfromtxt(path, delimiter=',')
    X = xy[:, :-1]
    y = xy[:, -1]
    return X, y


def load_data(path, file_type, max_data=0, max_dim=0,
              preprocess=True, include_offset=False, target_dim=None,
              pos_label=None):
    """Load data from a variety of file types.

    Parameters
    ----------
    path : string
        Data file path.

    file_type : string
        Supported file types are: 'svmlight', 'npy' (with the labels y in the
        rightmost col), 'npz', 'hdf5' (with datasets 'x' and 'y'), and 'csv'
        (with the labels y in the rightmost col)

    max_data : int
        If positive, maximum number of data points to use. If zero or negative,
        all data is used. Default is 0.

    max_dim : int
        If positive, maximum number of features to use. If zero or negative,
        all features are used. Default is 0.

    preprocess : boolean or Transformer, optional
        Flag indicating whether the data should be preprocessed. For sparse
        data, the features are scaled to [-1, 1]. For dense data, the features
        are scaled to have mean zero and variance one. Default is True.

    include_offset : boolean, optional
        Flag indicating that an offset feature should be added. Default is
        False.

    target_dim : int, optional
        When given, ensure X initially has this many features. Projection will
        be done after X is resized. Default is None.

    Returns
    -------
    X : array-like matrix, shape=(n_samples, n_features)

    y : int ndarray, shape=(n_samples,)
        Each entry indicates whether each example is negative (-1 value) or
        positive (+1 value)

    pp_obj : None or Transformer
        Transformer object used on data, or None if ``preprocess=False``
    """
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")

    if file_type in ["svmlight", "svm"]:
        X, y = _load_svmlight_data(path)
    elif file_type == "npy":
        X, y = _load_npy_data(path)
    elif file_type == "npz":
        X, y = _load_npz_data(path)
    elif file_type == "hdf5":
        X, y = _load_hdf5_data(path)
    elif file_type == "csv":
        X, y = _load_csv_data(path)
    else:
        raise ValueError("unsupported file type, %s" % file_type)


    if pos_label is None:
        y_vals = set(y)
        if len(y_vals) != 2:
            raise ValueError('Only expected y to take on two values, but instead'
                             'takes on the values ' + ', '.join(y_vals))
        if 1.0 not in y_vals:
            raise ValueError('y does not take on 1.0 as one on of its values, but '
                             'instead takes on the values ' + ', '.join(y_vals))
        if -1.0 not in y_vals:
            y_vals.remove(1.0)
            print('converting y values of %s to -1.0' % y_vals.pop())
            y[y != 1.0] = -1.0
    else:
        y[y != pos_label] = -1.0
        y[y == pos_label] = 1.0

    if preprocess is False:
        pp_obj = None
    else:
        if preprocess is True:
            if sp.issparse(X):
                pp_obj = preprocessing.MaxAbsScaler(copy=False)
            else:
                pp_obj = preprocessing.StandardScaler(copy=False)
        else:
            pp_obj = preprocess
            if target_dim is not None and target_dim != pp_obj.scale_.shape[0]:
                raise ValueError('target dim does not match pp_obj')
            target_dim = pp_obj.scale_.shape[0]
        if target_dim is not None:
            X_dim = X.shape[1]
            if X_dim < target_dim:
                print('expanding X')
                extra_shape = (X.shape[0], target_dim - X_dim)
                if sp.issparse(X):
                    stack_fun = sp.hstack
                    extra = sp.csr_matrix(extra_shape)
                else:
                    stack_fun = np.hstack
                    extra = np.zeros(extra_shape)
                X = stack_fun([X, extra])
            elif X_dim > target_dim:
                print('shrinking X')
                X = X[:,:target_dim]
        if preprocess is True:
            pp_obj.fit(X)
        X = pp_obj.transform(X)

    if include_offset:
        X = preprocessing.add_dummy_feature(X)

    if sp.issparse(X) and (X.nnz > np.prod(X.shape) / 10 or X.shape[1] <= 20):
        print("X is either low-dimensional or not very sparse, so converting "
              "to a numpy array")
        X = X.toarray()
    if isinstance(max_data, int) and max_data > 0 and max_data < X.shape[0]:
        X = X[:max_data,:]
        y = y[:max_data]
    if isinstance(max_dim, int) and max_dim > 0 and max_dim < X.shape[1]:
        X = X[:,:max_dim]

    return X, y, pp_obj


def _generate_and_save_from_X(X, theta, fname):
    lp = logistic_likelihood(theta, X, sum_result=False)
    ln = logistic_likelihood(theta, -X, sum_result=False)
    lmax = np.maximum(lp, ln)
    lp -= lmax
    ln -= lmax
    p = np.exp(lp) / (np.exp(lp) + np.exp(ln))
    y = npr.rand(X.shape[0])
    y[y <= p] = 1
    y[y != 1] = -1
    if fname is not None:
        if sp.issparse(X):
            save_sparse_Xy(fname, X, y)
        else:
            np.save(fname, np.hstack((X, y[:, np.newaxis])))
    return X, y


def _ensure_means_covar_match(means, covar):
    if len(means.shape) == 1:
        n_features = means.shape[0]
    else:
        n_features = means.shape[1]
    if len(covar.shape) != 2 or covar.shape[0] != covar.shape[1]:
        raise ValueError('invalid covariance matrix shape')
    if n_features != covar.shape[0]:
        raise ValueError('mean and covariance shapes do not match')


def generate_gaussian_synthetic(num_samples, mean, covar, theta,
                                fname=None, include_offset=False):
    """Generate classification data with covariates from Gaussian distribution.

    Generate `num_samples` data points with `X[i,:] ~ N(mean, covar)`, then use
    a logistic likelihood model with parameter `theta` to generate `y[i]`.
    If `include_offset = True`, then `X[i,-1] = 1`. Thus,
    `total_features = n_features` if `include_offset = False` and
    `n_features + 1` otherwise.

    Parameters
    ----------
    num_samples : int

    mean : array-like, shape=(n_features,)

    covar : matrix-like, shape=(n_features, n_features)

    theta : array-like, shape=(total_features,)

    fname : string, optional
        If provided, save data to the provided filename

    include_offset : boolean, optional
        Default is False.

    Returns
    -------
    X : ndarray with shape (num_samples, total_features)

    y : ndarray with shape (num_samples,)
    """
    _ensure_means_covar_match(mean, covar)
    X = npr.multivariate_normal(mean, covar, num_samples)
    if include_offset:
        X = np.hstack((X, np.ones((num_samples, 1))))
    return _generate_and_save_from_X(X, theta, fname)


def generate_gaussian_mixture(num_samples, weights, means, covar, theta,
                              fname=None, include_offset=False):
    """Generate classification data with covariates from Gaussian mixture.

    Generate `num_samples` data points with `X[i,:] ~ N(means[j,:], covar)`
    with probability `weights[j]`, then use a logistic likelihood model with
    parameter `theta` to generate `y[i]`.  If `include_offset = True`,
    then `X[i,-1] = 1`.  Thus, `total_features = n_features` if
    `include_offset = False` and `n_features + 1` otherwise.

    Parameters
    ----------
    num_samples : int

    weights : array-like, shape=(n_components,)

    means : array-like, shape=(n_components, n_features)

    covar : matrix-like, shape=(n_features, n_features)

    theta : array-like, shape=(total_features,)

    fname : string, optional
        If provided, save data to the provided filename

    include_offset : boolean, optional
        Default is False.

    Returns
    -------
    X : ndarray with shape (num_samples, total_features)

    y : ndarray with shape (num_samples,)
    """
    _ensure_means_covar_match(means, covar)
    if means.shape[0] != weights.shape[0]:
        raise ValueError("'means' and 'weights' shapes do not match")
    components = npr.choice(weights.shape[0], num_samples, p=weights)
    z = np.zeros(means.shape[1])
    X = means[components, :] + npr.multivariate_normal(z, covar, num_samples)
    if include_offset:
        X = np.hstack((X, np.ones((num_samples, 1))))
    return _generate_and_save_from_X(X, theta, fname)


def generate_reverse_mixture(num_samples, pos_prob, means, covar, fname=None):
    """Generate classification data class first, then Gaussian covariates.

    Generate `num_samples` data points with `Pr[y[i] = 1] = pos_prob` and
    `X[i,:] ~ N(means[y[i],:], covar)`.

    Parameters
    ----------
    num_samples : int

    pos_prob : float

    means : array-like, shape=(2, n_features)

    covar : matrix-like, shape=(n_features, n_features)

    fname : string, optional
        If provided, save data to the provided filename

    Returns
    -------
    X : ndarray with shape (num_samples, n_features)

    y : ndarray with shape (num_samples,)
    """
    _ensure_means_covar_match(means, covar)
    if means.shape[0] != 2:
        raise ValueError("'means' must have exactly two means")
    y = npr.rand(num_samples)
    y[y <= pos_prob] = 1
    y[y != 1] = -1
    components = np.zeros(num_samples, dtype=np.int)
    components[y == 1] = 1
    z = np.zeros(means.shape[1])
    X = means[components, :] + npr.multivariate_normal(z, covar, num_samples)
    if fname is not None:
        np.save(fname, np.hstack((X, y[:, np.newaxis])))
    return X, y


def generate_binary_data(num_samples, probs, theta,
                         fname=None, include_offset=False, ):
    """Generate classification data with binary covariates.

    Generate `num_samples` data points with `Pr[X[i,j] = 1] = probs[j]` and
    a logistic likelihood model with parameter `theta` to generate `y[i]`.
    If `include_offset = True`,  then `X[i,-1] = 1`.  Thus,
    `total_features = n_features` if `include_offset = False` and
    `n_features + 1` otherwise.

    Parameters
    ----------
    num_samples : int

    probs : array-like, shape=(n_features)

    theta : array-like, shape=(total_features,)

    fname : string, optional
        If provided, save data to the provided filename

    include_offset : boolean, optional
        Default is False.

    Returns
    -------
    X : csr_matrix with shape (num_samples, total_features)

    y : ndarray with shape (num_samples,)
    """
    probs = probs[np.newaxis, :]
    X = npr.rand(num_samples, probs.shape[1])
    X[X <= probs] = 1
    X[X != 1] = 0
    X = sp.csr_matrix(X, dtype=np.int32)
    if include_offset:
        X = sp.hstack((X, np.ones((num_samples, 1), dtype=np.int32)),
                      format='csr')
    return _generate_and_save_from_X(X, theta, fname)


def _process_row_entry(value, col_info, data, next_index):
    if col_info in ['lab', 'ign'] or value == '':
        return None, next_index
    elif col_info == 'num':
        if float(value) == 0:
            return None, next_index
        else:
            if '__index__' not in data:
                if next_index < 0:
                    return None, next_index
                data['__index__'] = next_index
                next_index += 1
            return (data['__index__'], value), next_index
    elif col_info == 'bin':
        if value not in data:
            if len(data) >= 2:
                raise ValueError('binary data column has more than two values')
            if len(data) == 0 or next_index < 0:
                data[value] = None
            else:
                data[value] = next_index
                next_index += 1
        if data[value] is None:
            return None, next_index
        else:
            return (data[value], "1"), next_index
    elif col_info == 'cat':
        if value not in data:
            if next_index < 0:
                data[value] = None
            else:
                data[value] = next_index
                next_index += 1
        if data[value] is None:
            return None, next_index
        else:
            return (data[value], "1"), next_index
    else:
        raise ValueError('invalid column info "%s"' % col_info)


def convert_categorical_data_to_svmlight(path, filetype, out_path, column_info,
                                         positive_labels,
                                         ignore_first_line=False,
                                         delimeter=',',
                                         init=None,
                                         no_new_features=False):
    """Convert categorical data into svmlight format.

    Column info is a space-separated list of information about each column.
    The options for each column are:
        * 'cat' - categorical data (induces multiple features)
        * 'bin' - binary data (induces single feature)
        * 'lab' - output label (can only be assigned to one column)
        * 'num' - numeric data
        * 'ign' - ignore column

    Parameters
    ----------
    path : string

    file_type : string
        Supported file types are: 'csv'

    out_path : string

    column_info : string

    positive_labels : list of strings

    ignore_first_line : boolean, optional
        Default is False.

    delimeter : string, optional
        Default is ','.

    init : tuple, optional
        Output from previous execution of the function. Used to maintain
        consistency across multiple conversions.

    no_new_features : boolean, optional
        If init is provided, then don't create any new features.

    Returns
    -------
    next_index : int

    data : object
    """
    info = column_info.split(' ')
    if info.count('lab') != 1:
        raise ValueError('column_info must specify exactly one label column')
    label_index = info.index('lab')
    if init is not None:
        next_index, data, label_map, next_label_id = init
        if no_new_features:
            next_index = -next_index
    else:
        next_index = 1
        data = [dict() for i in range(len(info))]
        next_label_id = 1
        label_map = {}

    if filetype == 'csv':
        with open(path, 'rb') as csv_file, open(out_path, 'wb') as out_file:
            reader = csv.reader(csv_file, delimiter=delimeter)
            try:
                if ignore_first_line:
                    reader.next()
                for row in reader:
                    if len(info) != len(row):
                        raise ValueError('row %d had an unexpected number of '
                                         'columns (expected %d, got %d)' %
                                         (reader.line_num, len(info), len(row)))
                    if positive_labels is None:
                        # hex_h = hashlib.md5(row[label_index]).hexdigest()
                        # h = int(hex_h, 16) % 49979687
                        # out_file.write('%d ' % h)
                        if row[label_index] not in label_map:
                            label_map[row[label_index]] = next_label_id
                            next_label_id += 1
                        out_file.write('%d ' % label_map[row[label_index]])
                    elif row[label_index] in positive_labels:
                        out_file.write('1 ')
                    else:
                        out_file.write('-1 ')
                    entry_list = []
                    for i, val in enumerate(row):
                        entry, next_index = _process_row_entry(val, info[i],
                                                               data[i],
                                                               next_index)
                        if entry is not None:
                            entry_list.append(entry)
                    entry_list.sort(cmp=lambda x,y: cmp(x[0], y[0]))
                    out_file.write(' '.join(['%s:%s' % e for e in entry_list]))
                    out_file.write('\n')
            except csv.Error as e:
                sys.exit('file %s, line %d: %s' % (path, reader.line_num, e))
        if len(label_map) > 0:
            with open(out_path + '.label_map', 'w') as f:
                cpk.dump(label_map, f)
        return abs(next_index), data
    else:
        raise ValueError("unsupported file type, %s" % file_type)
