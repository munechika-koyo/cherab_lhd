_cython_tree = declare_dependency(sources: [
  fs.copyfile('__init__.py'),
])

cython_sources = [
  ['emitters', 'emitters.pyx'],
]

foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    dependencies: [py_dep, np_dep],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/lhd/emc3/raytransfer',
  )
endforeach


python_sources = [
    '__init__.py',
    'raytransfer.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/emc3/raytransfer',
)
