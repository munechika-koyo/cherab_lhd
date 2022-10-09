"""
Module for tetrahedralization
"""
import numpy as np

from numpy cimport ndarray, uint32_t, import_array
cimport cython
from cython.parallel import prange

__all__ = ["tetrahedralize"]


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef ndarray[uint32_t, ndim=2] tetrahedralize(ndarray cells):
    """
    Generate Tetrahedral indices from cell indices array.
    One cubic-like cell is divided to 6 tetrahedra.

    :param ndarray cells: cell indices 2D array

    :return: tetrahedra indices array
    :rtype: ndarray
    """
    cdef:
        int i, j, k
        int[6][4] tetra_indices
        ndarray[uint32_t, ndim=2] tetrahedra
        uint32_t[:, ::1] tetrahedra_mv
        uint32_t[:, ::1] cells_mv

    if cells.ndim != 2:
        raise ValueError("cells must be a 2 dimensional array.")

    if cells.shape[1] != 8:
        raise ValueError("cells must have a shape of (N, 8).")

    # tetrahedra indices array
    tetrahedra = np.zeros((cells.shape[0] * 6, 4), dtype=np.uint32)

    # six tetrahedra indices at one cell
    tetra_indices[0][:] = [6, 2, 1, 0]
    tetra_indices[1][:] = [7, 3, 2, 0]
    tetra_indices[2][:] = [0, 7, 6, 2]
    tetra_indices[3][:] = [1, 5, 6, 4]
    tetra_indices[4][:] = [0, 4, 6, 7]
    tetra_indices[5][:] = [6, 4, 0, 1]

    # memory view
    tetrahedra_mv = tetrahedra
    cells_mv = cells

    for i in prange(cells_mv.shape[0], nogil=True):
        for j in range(6):
            for k in range(4):
                tetrahedra_mv[6 * i + j, k] = cells_mv[i, tetra_indices[j][k]]

    return tetrahedra