pyx_files = [
  ['emitters', 'emitters.pyx'],
]

foreach pyx_file : pyx_files
  py.extension_module(
    pyx_file[0],
    pyx_file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: py_dep,
    include_directories: [inc_np],
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
