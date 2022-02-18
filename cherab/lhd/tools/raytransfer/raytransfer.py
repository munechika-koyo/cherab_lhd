# This module offers the helper function to easily set raytransfer material

import os
import pickle
import numpy as np
from raysect.core.math import translate
from raysect.primitive import Cylinder
from cherab.lhd.emitter.E3E.geometry import EMC3
from cherab.lhd.tools.raytransfer.emitters import Discrete3DMeshRayTransferEmitter

# Constants
CHERAB_LHD_ROOT = os.path.join(os.path.abspath(__file__).split(sep="cherab_lhd")[0], "cherab_lhd")
INDEX_FUNC_PATH = os.path.join(CHERAB_LHD_ROOT, "output", "index_zones0-15.pickle")

RMIN, RMAX = 2.0, 5.5  # [m]
ZMIN, ZMAX = -1.6, 1.6


def load_rte_emc3(parent, path=None, bins=None):
    """
    Helper function of loding RayTransfer Emitter using :obj:`.Discrete3DMeshRayTransferEmitter`

    Parameters
    ----------
    parent : :obj:~`raysect.optical.world`
        raysect world Node
    path : str, optional
        path to picklized index function, by default cherab_lhd/output/index_zones0-15.pickle
    bins : int, optional
        the number of grids, by default ``14 * 18``

    Returns
    -------
    :obj:`~raysect.primitive.cylinder.Cylinder`
        primitive of cylinder
    """

    bins = bins or 14 * 18  # 14 zones x 18 degrees

    # Load index function
    index_func = load_index_function(path=path)

    # material as emitter
    material = Discrete3DMeshRayTransferEmitter(index_func, bins, integration_step=0.001)

    # primitive using cylinder
    shift = translate(0, 0, ZMIN)
    emitter = Cylinder(RMAX, ZMAX - ZMIN, transform=shift, parent=parent, material=material)

    return emitter


def load_index_function(path=None):
    """
    Load index function for Raytransfor Matrix

    Parameters
    ----------
    path : str
        path to picklized index function file, by default cherab_lhd/output/index_zones0-15.pickle

    Returns
    -------
    :obj:`.Discrete3DMesh`
        index function object
    """
    path = path or INDEX_FUNC_PATH
    with open(path, "rb") as f:
        index_func = pickle.load(f)

    return index_func


def create_14_zones():
    """ Create 14 zones using EMC3-EIRINE to facilitate the 3D tomography in LHD
    which returns a callable created by Discrete3DMesh class.
    """

    emc = EMC3()

    # load cell indeices mapping for tomography
    emc._phys_cells = mapping_14zones()

    # tetrahedralization
    emc.tetrahedralization()

    # create index function
    print("creating a geometric index function...")
    path = os.path.join(CHERAB_LHD_ROOT, "output", "index_zones0-15")
    func = emc.generate_index_function(path=path)

    return func


def mapping_14zones() -> dict:
    """
    Create 14 zones cell mapping table.
    This function returns mapping table which denotes the relationship
    between E3E's geometry cell index and 14 zones cell index for tomography.

    Returns
    -------
    dict{str: numpy.ndarray}
        each zones indices, e.g. {"zone0": array([0, 0, 0, ...])}
    """
    # define returns
    cell_map = {}

    # instatiate EMC3 class
    emc = EMC3()
    emc.load_grids()

    # -------------------------- zone0 -------------------------------
    zone = "zone0"

    # prepare indices data as teterahedra data
    tet_data = np.zeros(emc.num_cells[zone], dtype=np.uint32)

    offset = 0
    i = 0
    for i_tor in range(0, emc.num_toroidal[zone] - 1):

        if i_tor != 0 and i_tor % 4 == 0:
            offset += 14

        for i_pol in range(0, emc.num_poloidal[zone] - 1):
            for i_rad in range(0, emc.num_radial[zone] - 1):
                if i_rad < 4:
                    tet_data[i] = 0 + offset
                elif 4 <= i_rad < 8:
                    tet_data[i] = 1 + offset
                elif 8 <= i_rad < 12:
                    tet_data[i] = 2 + offset
                elif 12 <= i_rad < 16:
                    tet_data[i] = 3 + offset
                elif 16 <= i_rad:
                    if i_pol < 48:
                        tet_data[i] = 4 + offset
                    elif 48 <= i_pol < 170:
                        tet_data[i] = 5 + offset
                    elif 170 <= i_pol < 300:
                        tet_data[i] = 6 + offset
                    elif 300 <= i_pol < 430:
                        tet_data[i] = 7 + offset
                    elif 430 <= i_pol < 552:
                        tet_data[i] = 8 + offset
                    elif 552 <= i_pol:
                        tet_data[i] = 9 + offset

                i += 1

    cell_map[zone] = tet_data

    # ----------------------- zone1-4 --------------------------
    for zone, index in zip([f"zone{i}" for i in range(1, 5)], range(10, 14)):
        tet_data = np.zeros(emc.num_cells[zone], dtype=np.uint32)
        n = (emc.num_radial[zone] - 1) * (emc.num_poloidal[zone] - 1)

        for i_tor in range(0, emc.num_toroidal[zone] - 1, 4):
            tet_data[i_tor * n: (i_tor + 4) * n] = index
            index += 14
        cell_map[zone] = tet_data

    # ----------------------------------------------------------

    # ----------------------- zone11 ---------------------------
    zone = "zone11"

    # prepare indices data as teterahedra data
    tet_data = np.zeros(emc.num_cells[zone], dtype=np.uint32)

    offset = cell_map["zone4"][-1] + 1
    i = 0
    for i_tor in range(0, emc.num_toroidal[zone] - 1):

        if i_tor != 0 and i_tor % 4 == 0:
            offset += 14

        for i_pol in range(0, emc.num_poloidal[zone] - 1):
            for i_rad in range(0, emc.num_radial[zone] - 1):
                if i_rad < 4:
                    tet_data[i] = 0 + offset
                elif 4 <= i_rad < 8:
                    tet_data[i] = 1 + offset
                elif 8 <= i_rad < 12:
                    tet_data[i] = 2 + offset
                elif 12 <= i_rad < 16:
                    tet_data[i] = 3 + offset
                elif 16 <= i_rad < emc.num_radial[zone]:
                    if 459 <= i_pol < 547:
                        tet_data[i] = 4 + offset
                    elif 547 <= i_pol or i_pol < 53:
                        tet_data[i] = 5 + offset
                    elif 53 <= i_pol < 141:
                        tet_data[i] = 6 + offset
                    elif 141 <= i_pol < 249:
                        tet_data[i] = 7 + offset
                    elif 249 <= i_pol < 351:
                        tet_data[i] = 8 + offset
                    elif 351 <= i_pol < 459:
                        tet_data[i] = 9 + offset

                i += 1

    cell_map[zone] = tet_data

    next_index = cell_map["zone4"][-1] + 11
    # ----------------------- zone12-15 --------------------------
    for zone, index in zip([f"zone{i}" for i in range(12, 16)], range(next_index, next_index + 4)):
        tet_data = np.zeros(emc.num_cells[zone], dtype=np.uint32)
        n = (emc.num_radial[zone] - 1) * (emc.num_poloidal[zone] - 1)

        for i_tor in range(0, emc.num_toroidal[zone] - 1, 4):
            tet_data[i_tor * n: (i_tor + 4) * n] = index
            index += 14
        cell_map[zone] = tet_data

    # ----------------------------------------------------------

    return cell_map


if __name__ == "__main__":
    # emc = EMC3()
    # func = create_14_zones(emc)
    print("start debugging")
    cell_map = mapping_14zones()

    print("end debugging")
