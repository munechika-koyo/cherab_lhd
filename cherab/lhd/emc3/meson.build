# add include path to .pxd files
add_project_arguments(
  '-I',
  'cherab/lhd/emc3/cython/intfunction.pxd',
  '-I',
  'cherab/lhd/emc3/cython/discrete3dmesh.pxd',
  language : 'cython'
)


cython_sources = [
  ['geometry', 'geometry.pyx'],
]

foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: py_dep,
    include_directories: [inc_np],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/lhd/emc3',
  )
endforeach


python_sources = [
    '__init__.py',
    'barycenters.py',
    'curvilinear.py',
    'grid.py',
    'plasma.py'
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/emc3',
)


subdir('cython')
subdir('raytransfer')
subdir('repository')
subdir('inversion')
subdir('tests')
