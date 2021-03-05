import os
from math import ceil
import numpy as np

"""
This module provides useful functions for EMC3-EIRINE around file IO
"""

BASE = os.path.dirname(__file__)
GRID_PATH = os.path.join(BASE, "data", "grid-360.txt")


def read_E3E_grid(path=None):
    """
    Read EMC3-EIRINE grid data


    Parameters
    ----------
    path : str, optional
        path to the file discribing the E3E grid, by default grid-360.txt file stored in data directory

    Returns
    -------
    tuple
        r, z, num_rad, num_pol, num_tor, num_cells
        the type of r, z is dict containing grid coords data
        the type of num_rad, num_pol, num_tor is dict containing the number of grids
        associated with radial, poloidal and toroidal, respectively.
        the type of num_cells is the dict containing the number of cells in each zone
        ::
          >>> r, z, num_rad, num_pol, num_tor = read_E3E_grid()
          >>> r
          {'zone0': {0.0: [3.593351, 2.559212, ...],
                     0.25: [...],
                     :
                     9.0: [...]},
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

    # removing overlapping vertices in zone0 and zone11
    for zone in ["zone0", "zone11"]:
        if zone in zones:
            for tor_angle in r[zone].keys():
                r[zone][tor_angle][-num_rad[zone]:-1] = []
                z[zone][tor_angle][-num_rad[zone]:-1] = []
            num_pol[zone] -= 1

    return (r, z, num_rad, num_pol, num_tor, num_cells)


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
    if zone in ["zone0", "zone11"]:
        for i in range(start, start + N_rad - 1):
            faces += [(i, i + 1, i - start + 1, i - start)]
    return faces


def E3E_tetrahedralization(zone="zone1"):
    """
    generate tetrahedral vertices & indeces
    Dividing an E3E cell into five tetrahedra

    Parameters
    ----------
    zone : str, optional
        label of zone, by default "zone1"

    Returns
    -------
    numpy.ndarray (N, 3)
        (x, y, z) coords of tetrahedral vertices
    numpy.ndarray (M, 4)
        indeces of tetrahedral vertices
    """

    r, z, num_rad, num_pol, num_tor = read_E3E_grid()

    num_total = num_rad[zone] * num_pol[zone]
    vertices = np.zeros((num_total * num_tor[zone], 3), dtype=np.float64)
    offset = 0
    for tor_angle in r[zone].keys():
        tor_angle_rad = tor_angle * np.pi / 180.0
        for row in range(num_total):
            vertices[row + offset, :] = (r[zone][tor_angle][row] * np.cos(tor_angle_rad), r[zone][tor_angle][row] * np.sin(tor_angle_rad), z[zone][tor_angle][row])
        offset += num_total

    faces = generate_faces(num_rad, num_pol, zone=zone)

    tetrahedra = np.zeros((len(faces) * 5 * (num_tor[zone] - 1), 4), dtype=np.uint32)
    tet_id = 0
    offset = 0
    # Divide a cell into five tetrahedra.
    # Note that there are 2 ways of tetrahedralizatation of a E3E cell.
    # distinguish them using +,- sign
    tetra_type = int(+1)
    tetra_type_toroidal = int(+1)
    rad_index = 1

    for i in range(num_tor[zone] - 1):
        for face in faces:
            if rad_index > num_rad[zone] - 1:
                rad_index = 1
                if (num_rad[zone] - 1) % 2 == 0:
                    tetra_type *= -1

            if tetra_type > 0:
                # type 1 dividing tetra
                tetrahedra[tet_id, :] = np.array([face[0] + num_total, face[0], face[1], face[3]], dtype=np.uint32) + offset
                tetrahedra[tet_id + 1, :] = np.array([face[2] + num_total, face[1], face[2], face[3]], dtype=np.uint32) + offset
                tetrahedra[tet_id + 2, :] = np.array([face[1], face[1] + num_total, face[2] + num_total, face[0] + num_total], dtype=np.uint32) + offset
                tetrahedra[tet_id + 3, :] = np.array([face[3], face[3] + num_total, face[2] + num_total, face[0] + num_total], dtype=np.uint32) + offset
                tetrahedra[tet_id + 4, :] = np.array([face[1], face[3], face[0] + num_total, face[2] + num_total], dtype=np.uint32) + offset
            else:
                # type 2 dividing tetra
                tetrahedra[tet_id, :] = np.array([face[1] + num_total, face[0], face[1], face[2]], dtype=np.uint32) + offset
                tetrahedra[tet_id + 1, :] = np.array([face[3] + num_total, face[0], face[2], face[3]], dtype=np.uint32) + offset
                tetrahedra[tet_id + 2, :] = np.array([face[0], face[0] + num_total, face[1] + num_total, face[3] + num_total], dtype=np.uint32) + offset
                tetrahedra[tet_id + 3, :] = np.array([face[2], face[2] + num_total, face[3] + num_total, face[1] + num_total], dtype=np.uint32) + offset
                tetrahedra[tet_id + 4, :] = np.array([face[0], face[2], face[3] + num_total, face[1] + num_total], dtype=np.uint32) + offset
            tetra_type *= -1
            tet_id += 5
            rad_index += 1
        offset += num_total
        if tetra_type_toroidal > 0:
            # next toroidal face
            tetra_type = -1
        else:
            tetra_type = 1
        tetra_type_toroidal *= -1

    return vertices, tetrahedra


def E3E_tetrahedralization_six(zone="zone1"):
    """
    generate tetrahedral vertices & indeces
    Dividing an E3E cell into six tetrahedra

    Parameters
    ----------
    zone : str, optional
        label of zone, by default "zone1"

    Returns
    -------
    numpy.ndarray (N, 3)
        (x, y, z) coords of tetrahedral vertices
    numpy.ndarray (M, 4)
        indeces of tetrahedral vertices
    """

    r, z, num_rad, num_pol, num_tor = read_E3E_grid()

    num_total = num_rad[zone] * num_pol[zone]
    vertices = np.zeros((num_total * num_tor[zone], 3), dtype=np.float64)
    offset = 0
    for tor_angle in r[zone].keys():
        tor_angle_rad = tor_angle * np.pi / 180.0
        for row in range(num_total):
            vertices[row + offset, :] = (r[zone][tor_angle][row] * np.cos(tor_angle_rad), r[zone][tor_angle][row] * np.sin(tor_angle_rad), z[zone][tor_angle][row])
        offset += num_total

    faces = generate_faces(num_rad, num_pol, zone=zone)

    tetrahedra = np.zeros((len(faces) * 6 * (num_tor[zone] - 1), 4), dtype=np.uint32)
    tet_id = 0
    offset = 0

    # Divide a cell into six tetrahedra.

    for i in range(num_tor[zone] - 1):
        for face in faces:
            tetrahedra[tet_id, :] = np.array([face[0] + num_total, face[0], face[1], face[2]], dtype=np.uint32) + offset
            tetrahedra[tet_id + 1, :] = np.array([face[3] + num_total, face[3], face[0], face[2]], dtype=np.uint32) + offset
            tetrahedra[tet_id + 2, :] = np.array([face[1], face[1] + num_total, face[2] + num_total, face[0] + num_total], dtype=np.uint32) + offset
            tetrahedra[tet_id + 3, :] = np.array([face[2], face[2] + num_total, face[3] + num_total, face[0] + num_total], dtype=np.uint32) + offset
            tetrahedra[tet_id + 4, :] = np.array([face[0], face[2], face[3] + num_total, face[0] + num_total], dtype=np.uint32) + offset
            tetrahedra[tet_id + 5, :] = np.array([face[1], face[2], face[0] + num_total, face[2] + num_total], dtype=np.uint32) + offset

            tet_id += 6
        offset += num_total

    return vertices, tetrahedra


if __name__ == "__main__":
    # r, z, num_rad, num_pol, num_tor = read_E3E_grid()
    vertices, tetrahedra = E3E_tetrahedralization()
    pass
