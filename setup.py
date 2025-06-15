import os
import setuptools
import sys


class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__


def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

def requirements():
  with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'rt') as f:
    return f.readlines()

# NOTE: If cc3d.cpp does not exist:
# cython -3 --fast-fail -v --cplus cc3d.pyx

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++11', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++11', '-O3'
  ]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  name="connected-components-3d",
  version="3.24.0",
  setup_requires=['pbr', 'numpy', 'cython'],
  install_requires=['numpy'],
  python_requires=">=3.9,<4.0",
  extras_require={
    "stack": [ "crackle-codec", "fastremap" ],
  },
  ext_modules=[
    setuptools.Extension(
      'cc3d.fastcc3d',
      sources=[ 'cc3d/fastcc3d.pyx' ],
      language='c++',
      include_dirs=[ 'cc3d', str(NumpyImport()) ],
      extra_compile_args=extra_compile_args,
    )
  ],
  author="William Silversmith",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  package_data={
    'cc3d': [
      'COPYING',
      'COPYING.LESSER',
    ],
  },
  description="Connected components on discrete and continuous multilabel 3D and 2D images. Handles 26, 18, and 6 connected variants; periodic boundaries (4, 8, & 6).",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "LGPL-3.0-or-later",
  keywords = "connected-components CCL volumetric-data numpy connectomics image-processing biomedical-image-processing decision-tree union-find sauf 2d 3d",
  url = "https://github.com/seung-lab/connected-components-3d/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ],  
)


