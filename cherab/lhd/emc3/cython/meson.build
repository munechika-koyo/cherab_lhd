cython_sources = [
  ['intfunction', 'intfunction.pyx'],
  ['discrete3dmesh', 'discrete3dmesh.pyx'],
  ['mapper', 'mapper.pyx'],
  ['masking', 'masking.pyx'],
  ['tetrahedralization', 'tetrahedralization.pyx'],
]

# compile cython sources to .so modules
foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: [py_dep],
    include_directories: [inc_np],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/lhd/emc3/cython',
  )
endforeach


python_sources = [
    '__init__.py',
    '__init__.pxd',
    'discrete3dmesh.pxd',
    'intfunction.pxd',
    'mapper.pxd',
    'masking.pxd',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/emc3/cython',
)
