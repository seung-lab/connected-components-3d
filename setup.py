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

def configure_cpu_extension():
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

  return setuptools.Extension(
    'cc3d',
    sources=[ 'cc3d.pyx' ],
    language='c++',
    include_dirs=[ str(NumpyImport()) ],
    extra_compile_args=extra_compile_args,
  )

def configure_metal_gpu_extension():
  return setuptools.Extension(
      'cc3d_mps',
      sources=['src/gpu/cc3d_mps.pyx', 'src/gpu/cc3d_mps.mm'],
      extra_compile_args=['-ObjC++'],
      extra_link_args=['-framework', 'Metal', '-framework', 'MetalPerformanceShaders']
  )

extensions = [ configure_cpu_extension() ]

if sys.platform == 'darwin':
  extensions.append(configure_metal_gpu_extension())

setuptools.setup(
  name="connected-components-3d",
  version="3.17.0",
  setup_requires=['pbr', 'numpy', 'cython'],
  install_requires=['numpy'],
  python_requires=">=3.8,<4.0",
  ext_modules=extensions,
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
  license = "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
  keywords = "connected-components CCL volumetric-data numpy connectomics image-processing biomedical-image-processing decision-tree union-find sauf 2d 3d",
  url = "https://github.com/seung-lab/connected-components-3d/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ],  
)


