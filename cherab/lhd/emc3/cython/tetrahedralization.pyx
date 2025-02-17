"""Module for tetrahedralization."""

import numpy as np

cimport cython
from numpy cimport import_array, ndarray, uint32_t

from cython.parallel import prange

__all__ = ["tetrahedralize"]


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef ndarray[uint32_t, ndim=2] tetrahedralize(ndarray cells):
    """Generate tetrahedral indices from cell indices array.

    One cubic-like cell having 8 vertices is divided to 6 tetrahedra.

    Parameters
    ----------
    cells : (N, 8) ndarray
        Cell indices 2D array.

    Returns
    -------
    (6N, 4) ndarray
        Tetrahedra indices array.

    Examples
    --------
    >>> import numpy as np
    >>> from cherab.lhd.emc3.cython import tetrahedralize
    >>>
    >>> array = np.arange(16, dtype=np.uint32).reshape((2, -1))
    >>> array
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15]], dtype=uint32)
    >>> tetrahedralize(array)
    array([[ 6,  2,  1,  0],
           [ 7,  3,  2,  0],
           [ 0,  7,  6,  2],
           [ 1,  5,  6,  4],
           [ 0,  4,  6,  7],
           [ 6,  4,  0,  1],
           [14, 10,  9,  8],
           [15, 11, 10,  8],
           [ 8, 15, 14, 10],
           [ 9, 13, 14, 12],
           [ 8, 12, 14, 15],
           [14, 12,  8,  9]], dtype=uint32)
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
