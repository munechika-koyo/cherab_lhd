project(
  'cherab-lhd',
  'c',
  'cython',
  version: run_command(
    ['cherab/lhd/_build_utils/gitversion.py'],
    check: true
  ).stdout().strip(),
  meson_version: '>= 0.64.0',
  default_options: [
    'cython_args=-3',
    'c_std=c99',
    'buildtype=debugoptimized',
  ],
)

fs = import('fs')

# https://mesonbuild.com/Python-module.html
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# OpenMP dependency
omp_dep = dependency('openmp', required: true)


# NumPy include directory - needed in all submodules
incdir_numpy = run_command(
  py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

# Numpy deprecated api (if errors is occurred, comment out in module section)
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# subdirectories
subdir('cherab/lhd')
