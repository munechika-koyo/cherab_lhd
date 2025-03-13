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
    'c_std=c17',
    'buildtype=debugoptimized',
  ],
)

cc = meson.get_compiler('c')
fs = import('fs')
py = import('python').find_installation(pure: false)

# Python dependency
py_dep = py.dependency()

# OpenMP dependency
omp_dep = dependency('openmp', required: false)
if not omp_dep.found()
  if host_machine.system() == 'darwin'
    dirs = run_command('brew', ['--prefix'], check: true).stdout().strip() + '/lib'
    message('Looking for OpenMP in ' + dirs)
    omp_dep = cc.find_library('libomp', dirs: dirs, required: true)
  else
    error('OpenMP is required for this project')
  endif
endif

# NumPy dependency
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()

  # We do need an absolute path to feed to `cc.find_library` below
  _incdir_numpy_abs = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)
# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released versions
# when NumPy introduces a new deprecation.
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)

# subdirectories
subdir('cherab')
