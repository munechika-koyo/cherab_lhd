pyx_files = [
  ['samplers', 'samplers.pyx'],
]

foreach pyx_file : pyx_files
  py.extension_module(
    pyx_file[0],
    pyx_file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: py_dep,
    include_directories: [inc_np],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/lhd/tools',
  )
endforeach


python_sources = [
    '__init__.py',
    '__init__.pxd',
    'spinner.py',
    'visualization.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/tools',
)
