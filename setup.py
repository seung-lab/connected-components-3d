import os
import setuptools

join = os.path.join

# NOTE: If cc3d.cpp does not exist:
# cython -3 --fast-fail -v --cplus cc3d.pyx

import numpy as np

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  install_requires=['numpy'],
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  ext_modules=[
    setuptools.Extension(
      'cc3d',
      sources=[ 'cc3d.cpp' ],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=[
        '-std=c++11', '-O3', '-ffast-math'
      ]
    )
  ],
  pbr=True)





