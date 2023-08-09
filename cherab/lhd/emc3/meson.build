# add include path to .pxd files
add_project_arguments(
  '-I',
  'cherab/lhd/emc3/cython/intfunction.pxd',
  '-I',
  'cherab/lhd/emc3/cython/discrete3dmesh.pxd',
  language : 'cython'
)


pyx_files = [
  ['geometry', 'geometry.pyx'],
]

foreach pyx_file : pyx_files
  py.extension_module(
    pyx_file[0],
    pyx_file[1],
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
subdir('inversions')
