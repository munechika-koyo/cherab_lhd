cython_sources = [
  ['pinhole', 'pinhole.pyx'],
]

foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: py_dep,
    include_directories: [inc_np],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/lhd/observer/imaging',
  )
endforeach


python_sources = [
    '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/observer/imaging',
)
