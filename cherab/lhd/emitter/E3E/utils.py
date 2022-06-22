import os
from math import ceil
import pickle
import numpy as np

"""
This module provides useful functions for EMC3-EIRINE around file IO
"""

BASE = os.path.dirname(__file__)
GRID_PATH = os.path.join(BASE, "data", "grid-360.txt")
CELLGEO_PATH = os.path.join(BASE, "data", "CELL_GEO")


def read_E3E_grid(path=None, save=False) -> tuple:
    """
    Read EMC3-EIRINE grid data. The grid position data (r, z) in each toroidal angle
    and zone are written in grid-*.txt. This function allows to read them and return variables
    one of which is dict containing (r, z, phi) ndarray which each zone's key and some numbers (num_rad, num_pol, num_tor, num_cells).

    Parameters
    ----------
    path : str, optional
        path to the file discribing the E3E grid, by default grid-360.txt file stored in data directory
    save : bool, optional
        the flag whether grids and numbers are stored as binary file with pickeling, by default False.
        The path to save binary file is same as a grid text file.

    Returns
    -------
    tuple
        (grids, num_rad, num_pol, num_tor, num_cells)
        the type of grids is dict containing grid coords data as numpy.ndarray.
        the type of num_rad, num_pol, num_tor is dict containing the number of grids
        associated with radial, poloidal and toroidal, respectively.
        the type of num_cells is the dict containing the number of cells in each zone
        ::
          >>> grids, num_rad, num_pol, num_tor = read_E3E_grid(save=False)
          >>> grids
          {'zone0': numpy.ndarray (N, 3, M) # array[:, :, i] denotes coords. (r, z, phi) in a polidal plane,
           'zone1': ...,
           :
           'zone21': ...
           }
           >>> num_rad
           {'zone0': 81, 'zone1': 97, 'zone2': 97, 'zone3': 97, 'zone4': 97, ...}
    """

    # path to E3E grid data text file & type checking
    path = path or GRID_PATH
    zones = [f"zone{i}" for i in range(0, 21 + 1)]

    # open text file
    with open(path, "r") as f:
        # Define variables
        r = {zone: {} for zone in zones}
        z = {zone: {} for zone in zones}
        num_rad = {key: 0 for key in r.keys()}
        num_pol = {key: 0 for key in r.keys()}
        num_tor = {key: 0 for key in r.keys()}
        num_cells = {key: 0 for key in r.keys()}

        # Load each zone points
        for zone in zones:

            # Load  Number of (radial, poloidal, toroidal) points
            line = f.readline()
            num_rad[zone], num_pol[zone], num_tor[zone] = [int(x) for x in line.split()]
            num_row = ceil(num_rad[zone] * num_pol[zone] / 6)  # row number per r/z points

            for _ in range(num_tor[zone]):
                # Load toroidal angle
                line = f.readline()
                tor_angle = float(line)

                # Load (r, z) point on a polidal plane
                r[zone][tor_angle] = []
                z[zone][tor_angle] = []
                for _ in range(num_row):
                    line = f.readline()
                    r[zone][tor_angle] += [float(x) * 1.0e-2 for x in line.split()]  # [cm] -> [m]
                for _ in range(num_row):
                    line = f.readline()
                    z[zone][tor_angle] += [float(x) * 1.0e-2 for x in line.split()]  # [cm] -> [m]
                line = f.readline()  # for removing only "\n" line

            # counting the number of cells in each zone
            num_cells[zone] = (num_rad[zone] - 1) * (num_pol[zone] - 1) * (num_tor[zone] - 1)

    # convert grid coords list (r, z) into 3-dim numpy.ndarray (r, z, phi)
    # and vaeiable grids has zones key "zone0", "zone1", ...
    grids = {zone: None for zone in zones}
    for zone in grids.keys():
        grids[zone] = np.zeros((num_rad[zone] * num_pol[zone], 3, num_tor[zone]), dtype=np.float64)
        for i, phi in enumerate(r[zone].keys()):
            grids[zone][:, :, i] = np.array([r[zone][phi],
                                             z[zone][phi],
                                             [phi] * len(z[zone][phi])]).T

    if save:
        save_path = os.path.splitext(path)[0] + ".pickle"
        with open(save_path, "wb") as f:
            pickle.dump((grids, num_rad, num_pol, num_tor, num_cells), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved grids to {save_path} successfully.")

    return (grids, num_rad, num_pol, num_tor, num_cells)


def read_cell_index(path=None, save=False) -> np.ndarray:
    """Read cell geometry indices from Raw text file.
    EMC3 has numerous geometric cells, the number of which is defined :math:`num_cells` in each zones.
    However, EMC3 calculates the physical values such as plasma density in each cell combined several geometric cells,
    which is called "physical" cells. Their relationship between geomtric and physical cells indices is written in CELL_GEO file.

    Parameters
    ----------
    path : str, optional
        path to the CELL_GEO file, by default ".../data/CELL_GEO"
        The default file "CELL_GEO" has geometric indices in all zones.
    """
    # path to CELL_GEO file
    path = path or CELLGEO_PATH

    # load cell_geo indices
    with open(path, "r") as f:
        line = f.readline()
        num_geo, num_plasma, num_total = list(map(int, line.split()))
        line = f.readline()
        cell_index = np.array([int(x) - 1 for x in line.split()], dtype=np.uint32)  # for c language index format

    if save:
        save_path = os.path.splitext(path)[0] + ".pickle"
        with open(save_path, "wb") as f:
            pickle.dump(cell_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved grids to {save_path} successfully.")

    return cell_index


if __name__ == "__main__":
    # grids, num_rad, num_pol, num_tor, num_cells = read_E3E_grid(save=False)
    cell_index = read_cell_index(save=False)
    pass
