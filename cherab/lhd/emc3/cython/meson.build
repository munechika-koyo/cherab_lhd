pyx_files = [
  ['intfunction', 'intfunction.pyx'],
  ['discrete3dmesh', 'discrete3dmesh.pyx'],
  ['mapper', 'mapper.pyx'],
  ['masking', 'masking.pyx'],
  ['tetrahedralization', 'tetrahedralization.pyx'],
]

# compile .pyx -> .so
foreach pyx_file : pyx_files
  py.extension_module(
    pyx_file[0],
    pyx_file[1],
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
