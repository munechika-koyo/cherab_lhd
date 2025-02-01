"""This script installs EMC3 data into the specified HDF5 storage.

The script takes the following command-line arguments:
- --data-dir: Directory containing the data files (required)
- --grid-filename: Filename for the grid data (required)
- --cell-filename: Filename for the cell data (required)
- --overwrite: Flag to overwrite existing data (optional)

The script performs the following steps:
1. Parses the input parameters.
2. Installs the grid data into the HDF5 storage.
3. Installs the physical cell indices into the HDF5 storage.
4. Installs the cell indices into the HDF5 storage.
5. Installs the remaining data into the HDF5 storage.

Usage:
    pixi python install_data.py --data-dir <data_directory> --grid-filename <grid_file> --cell-filename <cell_file> [--overwrite]

Example:
    pixi python install_data.py --data-dir /path/to/data --grid-filename grid.h5 --cell-filename cell.h5 --overwrite
"""

import argparse
from pathlib import Path

from cherab.lhd.emc3.repository.install import (
    install_cell_indices,
    install_data,
    install_grids,
    install_physical_cell_indices,
)
from cherab.lhd.tools.fetch import PATH_TO_STORAGE

# Parse input parameters
parser = argparse.ArgumentParser(description="Install EMC3 data")
parser.add_argument("--data-dir", required=True, help="Directory containing the data files")
parser.add_argument("--grid-filename", required=True, help="Filename for the grid data")
parser.add_argument("--cell-filename", required=True, help="Filename for the cell data")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data")
args = parser.parse_args()

data_dir = Path(args.data_dir)
grid_filename = args.grid_filename
cell_filename = args.cell_filename
overwrite = args.overwrite

# install grids
install_grids(
    data_dir / grid_filename,
    hdf5_path=PATH_TO_STORAGE / "emc3.hdf5",
    update=overwrite,
)
install_physical_cell_indices(
    data_dir / cell_filename,
    hdf5_path=PATH_TO_STORAGE / "emc3.hdf5",
    update=overwrite,
)
install_cell_indices(hdf5_path=PATH_TO_STORAGE / "emc3.hdf5", update=overwrite)
install_data(data_dir, hdf5_path=PATH_TO_STORAGE / "emc3.hdf5")
