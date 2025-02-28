_cython_tree = declare_dependency(sources: [
  fs.copyfile('__init__.pxd'),
  fs.copyfile('__init__.py'),
  fs.copyfile('discrete3dmesh.pxd'),
  fs.copyfile('intfunction.pxd'),
  fs.copyfile('mapper.pxd'),
  fs.copyfile('masking.pxd'),
  fs.copyfile('tetrahedralization.pxd'),
  fs.copyfile('tetra_mesh.pxd'),
])

cython_sources = [
  ['intfunction', 'intfunction.pyx'],
  ['discrete3dmesh', 'discrete3dmesh.pyx'],
  ['mapper', 'mapper.pyx'],
  ['masking', 'masking.pyx'],
  ['tetrahedralization', 'tetrahedralization.pyx'],
  ['utility', 'utility.pyx'],
  ['tetra_mesh', 'tetra_mesh.pyx'],
]

# compile cython sources to .so modules
foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    dependencies: [py_dep, np_dep, omp_dep, _cython_tree],
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
    'tetrahedralization.pxd',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/emc3/cython',
)
