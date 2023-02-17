python_sources = [
    'format_cad_file.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/lhd/machine/geometry',
)


install_subdir(
  'data',
  exclude_directories: ['data/STLfiles'],
  install_dir: 'cherab/lhd/machine/geometry',
  strip_directory: false
)
