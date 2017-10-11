from distutils.core import setup
from distutils.extension import Extension
import numpy as np

from Cython.Build import cythonize

extensions = [
    Extension("*", ["passglm/*.pyx"],
              include_dirs = [np.get_include()],
              ),
]

setup(
    name = 'passglm',
    version='0.1',
    description="PASS-GLM.",
    author='Jonathan H. Huggins',
    author_email='jhuggins@mit.edu',
    url='https://bitbucket.org/jhhuggins/pass-glm/',
    packages=['passglm'],
    package_data={'approxss' : ['*.so']},
    install_requires=[
        'Cython >= 0.20.1', 'numpy', 'scipy', 'matplotlib', 'ray',
        'sklearn', 'h5py', 'seaborn', 'nose', 'future'],
    ext_modules = cythonize(extensions),
    keywords = ['Bayesian', 'GLMs', 'scalable inference',
                'sufficient statistics'],
    platforms='ALL',
)
