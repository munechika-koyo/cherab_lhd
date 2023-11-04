cython_sources = [
  ['polygon', 'polygon.pyx'],
]

foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    c_args: [numpy_nodepr_api],
    dependencies: py_dep,
    include_directories: [inc_np],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/lhd/emc3/inversion',
  )
endforeach


python_sources = [
    '__init__.py',
    'derivative.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/emc3/inversion',
)

subdir('tests')
