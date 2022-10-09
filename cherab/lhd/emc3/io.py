"""
Module to provide useful functions around EMC3-EIRENE file IO
"""
from __future__ import annotations

import json
import os
from math import ceil

import numpy as np
from cherab.core.utility import RecursiveDict

__all__ = ["grid2array", "physical_index2array"]


BASE = os.path.dirname(__file__)
GRID_PATH = os.path.join(BASE, "data", "grid-360", "grid-360.txt")
CELLGEO_PATH = os.path.join(BASE, "data", "grid-360", "CELL_GEO")


def grid2array(path: str = GRID_PATH) -> None:
    """
    Convert raw grid text data into numpy 3-D ndarray binary file.
    EMC3-EIRENE has several comutational area (which is called "zone").
    Each zone is defined by grids of :math:`(R, Z)` coordinates at each toroidal angle :math:`\\phi`,
    which are recorded in ``grid-*.txt`` file. This function allows to read and parse them, and
    returns 3-D ndarrays containing :math:`(R, Z, \\phi)` coordinates at one poloidal plane in one
    dimension.
    This function also generates ``grid_config.json`` file into the same path's directory.
    In this json file, the number of grid resolution, cells in each zone is recorded.

    Parameters
    ----------
    path
        path to the raw text file by default ``./data/grid/grid-360.txt``
    """

    # check if file path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exists.")

    # zone labels, by default zone0 ~ zone21
    zones = [f"zone{i}" for i in range(0, 21 + 1)]

    # grid config dictionary
    grid_config = RecursiveDict()

    # define number of grid resolution
    L: int  # Radial grid resolution
    M: int  # Poloidal grid resolution
    N: int  # Toroidal grid resolution

    # Parse text to grid coordinates
    with open(path, "r") as file:

        # Load each zone area
        for zone in zones:

            # Parse number of grid resolution
            line = file.readline()
            L, M, N = [int(x) for x in line.split()]
            num_rows = ceil(L * M / 6)  # number of table rows per r/z points

            # grid array
            grid = np.zeros((L * M, 3, N), dtype=np.float64)

            for n in range(N):
                # Parse toroidal angle
                line = file.readline()
                toroidal_angle = float(line)

                # Load (r, z) point on a polidal plane
                r_coords: list[float] = []
                z_coords: list[float] = []

                for _ in range(num_rows):
                    line = file.readline()
                    r_coords += [float(x) * 1.0e-2 for x in line.split()]  # [cm] -> [m]
                for _ in range(num_rows):
                    line = file.readline()
                    z_coords += [float(x) * 1.0e-2 for x in line.split()]  # [cm] -> [m]

                line = file.readline()  # to skip one line

                # store coordinates into 3-D ndarray (r, z, phi)
                grid[:, 0, n] = r_coords
                grid[:, 1, n] = z_coords
                grid[:, 2, n] = np.repeat(toroidal_angle, L * M)

            # save grid data as .npy file
            filename = os.path.join(os.path.dirname(path), f"grid-{zone}.npy")
            np.save(filename, grid)

            # store grid config
            grid_config[zone]["L"] = L
            grid_config[zone]["M"] = M
            grid_config[zone]["N"] = N
            grid_config[zone]["num_cells"] = (L - 1) * (M - 1) * (N - 1)

    # save grid config as json format
    json_path = os.path.join(os.path.dirname(path), "grid_config.json")
    with open(json_path, "w") as file:
        json.dump(grid_config.freeze(), file, indent=4)


def physical_index2array(path: str = CELLGEO_PATH) -> None:
    """
    Convert EMC3-EIRENE physical cell indices stored in raw text file into numpy ndarray.
    EMC3 has numerous geometric cells, each of which forms a cubic-like shape with 8 vetices
    in each zones. To identify cells, an index number, which is called 'physical index' is allocated
    to each cell. Here such indices are parsed from text file and save them into numpy ndarray
    binary file in each zone.

    .. note::

        Before running this function, make sure if ``grid_config.json`` file exists in same path's
        directory. The number of cells in each zone is required.

    Parameters
    ----------
    path
        path to the raw text file: CELL_GEO, by default ``".../data/grid/CELL_GEO"``
        The default ``CELL_GEO`` file has geometric indices in all zones.
    """
    # check if file path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exists.")

    # check if grid_config.json file exists
    grid_config_path = os.path.join(os.path.dirname(path), "grid_config.json")
    if not os.path.exists(grid_config_path):
        raise FileNotFoundError(f"{grid_config_path} does not exists.")

    # load grid config
    with open(grid_config_path, "r") as file:
        grid_config = json.load(file)

    # load cell indices
    # starting from zero for c language index format
    indices = np.loadtxt(path, dtype=np.uint32, skiprows=1) - 1

    # save indices into each .npy file
    start = 0
    for zone in grid_config.keys():
        num_cells = grid_config[zone]["num_cells"]
        filename = os.path.join(os.path.dirname(path), f"indices-{zone}.npy")
        np.save(filename, indices[start : start + num_cells])
        start += num_cells


if __name__ == "__main__":
    grid2array()
    # physical_index2array()
    pass
