# passglm: a package for creating and evaluating PASS-GLM models

The `passglm` package was used produce the experiments for:

[Jonathan H. Huggins](http://www.jhhuggins.org),
[Ryan P. Adams](http://www.cs.princeton.edu/~rpa/),
[Tamara Broderick](http://www.tamarabroderick.com).
*[PASS-GLM: polynomial approximate sufficient statistics for scalable Bayesian GLM inference ](https://arxiv.org/abs/1709.09216)*.
In *Proc. of the 31st Annual Conference on Neural Information Processing
Systems* (NIPS), 2017.

The package includes functionality to load data, construct PASS-GLM approximations
for logistic regression, run an adaptive Metropolis-Hastings sampler, and
compare performance of PASS-GLM inferences to those obtained with
other methods.
Support for streaming and distributed inference is included.

## Compilation and testing

To compile and test the package (for development purposes):
```bash
python setup.py build_ext --inplace  # compile cython code in place
nosetests tests/                     # run tests, which takes a minute or so
```

To install:
```bash
pip install .
```

## Usage

For example usages, see the [scripts/](scripts/) directory.
