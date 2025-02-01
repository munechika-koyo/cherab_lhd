cimport numpy as np
from numpy cimport import_array

import_array()


cdef struct GridConfig:
    int L  # Radial grid resolution
    int M  # Poloidal grid resolution
    int N  # Toroidal grid resolution
    int num_cells  # Number of cells


cdef class Grid:

    cdef:
        str _zone
        str _grid_group
        tuple[int, int, int] _shape
        object _hdf5_path
        GridConfig _config
        np.ndarray _grid_data

    cpdef np.ndarray generate_vertices(self)

    cpdef np.ndarray generate_cell_indices(self)
