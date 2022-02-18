# This module offers the helper function to easily set raytransfer material

import os
import pickle
import numpy as np
from numpy.lib.arraysetops import isin
from cherab.lhd.emitter.E3E.geometry import EMC3
from cherab.lhd.tools.raytransfer import mapping_14zones

# Constants
CHERAB_LHD_ROOT = os.path.join(os.path.abspath(__file__).split(sep="cherab_lhd")[0], "cherab_lhd")


def mapping_zones14_to_phys_cell():
    # define returns
    mapping = {key: set() for key in range(0, 14 * 18)}

    # load phys_cell_mapping
    emc = EMC3()
    emc.load_cell_index()

    # load 14 zones cell mapping
    zones14_mapping = mapping_14zones()

    # create mapping table
    for zone in emc.zones:
        for key, element in zip(zones14_mapping[zone], emc.phys_cells[zone]):
            mapping[key].add(element)

    return mapping


def resize_E3E_data(data, mapping=None):
    """resizing EMC3-EIRENE data into 14 zones meshes by summing up several values.

    Parameters
    ----------
    data : numpy.ndarray
        EMC3-EIRENE 1D-array data

    mapping : dict, optional
        mapping dictionarry to map tomographic cell index into physical one.
        If None, mapping_zones14_to_phys_cell() is automatically called.

    Returns
    -------
    numpy.ndarray
        resized data array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be numpy.ndarray type.")

    if not mapping:
        mapping = mapping_zones14_to_phys_cell()

    resized_data = np.zeros(14 * 18, dtype=np.float64)
    for key, value in mapping.items():
        resized_data[key] = np.mean([data[i] for i in value if i < data.size] or 0.0)

    return resized_data


if __name__ == "__main__":
    # emc = EMC3()
    # func = create_14_zones(emc)
    cell_map = mapping_zones14_to_phys_cell()

    pass
