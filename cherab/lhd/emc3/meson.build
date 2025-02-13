_cython_tree = declare_dependency(sources: [
  fs.copyfile('__init__.py'),
])

cython_sources = [
  ['geometry', 'geometry.pyx'],
]

foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    dependencies: [py_dep, np_dep, omp_dep, _cython_tree],
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
