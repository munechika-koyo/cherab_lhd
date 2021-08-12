import os
from math import ceil
import pickle
import numpy as np

"""
This module provides useful functions for EMC3-EIRINE around file IO
"""

BASE = os.path.dirname(__file__)
GRID_PATH = os.path.join(BASE, "data", "grid-360.txt")


def read_E3E_grid(path=None, save=False):
    """
    Read EMC3-EIRINE grid data. The grid position data (r, z) in each toroidal angle
    and zone are written in grid-***.txt. This function allows to read them and return variables
    one of which is dict contining (r, z, phi) ndarray which each zone's key and some numbers (num_rad, num_pol, num_tor, num_cells).

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
        grids[zone] = np.zeros((num_rad[zone] * num_pol[zone], 3, num_tor[zone]))
        for i, phi in enumerate(r[zone].keys()):
            grids[zone][:, :, i] = np.array([r[zone][phi],
                                             z[zone][phi],
                                             [phi] * len(z[zone][phi])]).T

    if save:
        save_path = os.path.splitext(path)[0] + ".pickel"
        with open(save_path, "wb") as f:
            pickle.dump((grids, num_rad, num_pol, num_tor, num_cells), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved grids to {save_path} successfully.")

    return (grids, num_rad, num_pol, num_tor, num_cells)


def generate_faces(num_rad, num_pol, zone="zone0"):
    """generate cell indeces

    Parameters
    ----------
    num_rad : dict
        the number of grids along the radial direction
    num_pol : dict
        the number of grids along the poloidal direction
    zone : str, optional
        label of zones, by default "zone0"

    Returns
    -------
    list
        containing face indeces
    """
    faces = []
    start = 0
    N_rad = num_rad[zone]
    N_pol = num_pol[zone]

    while start < N_rad * (N_pol - 1):
        for i in range(start, start + N_rad - 1):
            faces += [(i, i + 1, i + 1 + N_rad, i + N_rad)]
        start += N_rad

    return faces


if __name__ == "__main__":
    grids, num_rad, num_pol, num_tor, num_cells = read_E3E_grid(save=False)
    pass
