cimport numpy as np
from numpy cimport import_array

import_array()

cdef struct GridConfig:
    int L  # Radial grid resolution
    int M  # Poloidal grid resolution
    int N  # Toroidal grid resolution
    int total  # Total number of grid points


cdef class CenterGrids:

    cdef:
        str _zone
        str _grid_group
        str _index_type
        tuple[int, int, int] _shape
        object _hdf5_path
        GridConfig _config
        np.ndarray _grid_data

    cpdef (int, int, int) get_lmn(self, int index)
