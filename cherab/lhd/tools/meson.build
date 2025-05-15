cython_sources = [
  ['samplers', 'samplers.pyx'],
]

foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    dependencies: [py_dep, np_dep],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/lhd/tools',
  )
endforeach

python_sources = [
  '__init__.py',
  'spinner.py',
  'fetch.py',
  'visualization.py',
  'utility.py',
  'registries.json',
  'urls.json',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/tools',
)
