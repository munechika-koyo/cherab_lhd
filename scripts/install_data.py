"""This script installs EMC3 raw data into the user's cache.

The script takes the following command-line arguments:
- --data-dir: Directory containing the data files (required)
- --grid-filename: Filename for the grid data (required)
- --cell-filename: Filename for the cell data (required)

Examples
--------
.. code-block:: bash

    pixi run python install_data.py --data-dir /path/to/data --grid-filename grid-360.txt --cell-filename CELL_GEO
"""

import argparse
import json
from importlib.resources import as_file, files
from pathlib import Path

from pooch import file_hash

from cherab.lhd.emc3.repository.install import (
    install_center_points,
    install_data,
    install_grids,
    install_indices,
)
from cherab.lhd.tools.fetch import PATH_TO_STORAGE, get_registries

# Parse input parameters
parser = argparse.ArgumentParser(description="Install EMC3 data")
parser.add_argument("--data-dir", required=True, help="Directory containing the data files")
parser.add_argument("--grid-filename", required=True, help="Filename for the grid data")
parser.add_argument("--cell-filename", required=True, help="Filename for the cell data")
args = parser.parse_args()

data_dir = Path(args.data_dir)
grid_filename = Path(args.grid_filename)
cell_filename = Path(args.cell_filename)

# Install grids. NOTE: must be done first
install_grids(data_dir / grid_filename)

# Install pre-configured indices
install_indices(data_dir / cell_filename)

# Install center points
install_center_points()

# Install remaining data
install_data(data_dir)

# Update registry keys
key_registry = f"emc3/{grid_filename.with_suffix('.nc')}"
hash = file_hash(PATH_TO_STORAGE / key_registry)
registry = get_registries()
registry.update({key_registry: hash})
with as_file(files("cherab.lhd.tools").joinpath("registries.json")) as file:
    with file.open("w") as f:
        json.dump(registry, f, indent=2)
