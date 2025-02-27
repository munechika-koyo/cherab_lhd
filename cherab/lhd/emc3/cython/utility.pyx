"""Module to provide utility functions for the EMC3 package."""

cimport cython
cimport numpy as np
from numpy cimport import_array

from .tetrahedralization cimport tetrahedralize

import numpy as np

from cython.parallel import prange


__all__ = ["compute_centers"]

import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef np.ndarray compute_centers(
    double[:, ::1] vertices, np.ndarray cells, np.ndarray indices
):
    """Compute center points of each cell.

    Assume that each cell is formed by combining an identical number of sub-cells as specified
    by the indices array.
    Center points are calculated by averaging the vertices of each sub-cell,
    which are regarded as the average of barycenters of six tetrahedra.

    .. note::

        The overall resolution must be preserved when combining cells; localized variations in
        resolution are not permitted.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Grid vertices. N is the number of vertices, and 3 is the number of coordinates
        :math:`(X, Y, Z)`.
    cells : (M, 8) ndarray
        Grid cells. M is the number of cells, and 8 is the number of vertices for each cell.
    indices : (L', M', N') ndarray[uint32]
        Specific index array.

    Returns
    -------
    (L'', M'', N'', 3) ndarray
        Center points array with new resolution.
        Each axis corresponds to the radial, poloidal, toroidal, and XYZ coordinates, respectively.

    Examples
    --------
    >>> import numpy as np
    >>> verts = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
    ...                   [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
    >>> cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint32)
    >>> indices = np.array([[[0]]], dtype=np.uint32)  # shape (1, 1, 1)
    >>> compute_centers(verts, cells, indices)
    array([[[[0.5, 0.5, 0.5]]]])
    """
    cdef:
        np.ndarray[np.uint32_t, ndim=2] tetrahedra
        np.uint32_t[:, ::1] tetrahedra_mv
        np.ndarray[np.float64_t, ndim=2] verts, verts_rec
        double[:, ::1] verts_mv, verts_rec_mv
        np.uint32_t[:, :, :] indices_mv
        np.uint32_t[::1] count_array_mv
        int num_cell, i, j, k, l, m, n, L, M, L_new, M_new, N_new, total, index

    indices_mv = indices

    # Create cells and tetrahedra from Grid
    tetrahedra = tetrahedralize(cells.astype(np.uint32))
    tetrahedra_mv = tetrahedra

    num_cell = cells.shape[0]

    # Calculate center of each cell (with original resolution)
    verts = np.zeros((num_cell, 3), dtype=float)
    verts_mv = verts

    # With divide by 6 x 4 for one cell
    for i in prange(num_cell, nogil=True):
        for j in range(6 * i, 6 * (i + 1)):
            for k in tetrahedra_mv[j, :]:
                verts_mv[i, 0] += vertices[k, 0] / 24.0
                verts_mv[i, 1] += vertices[k, 1] / 24.0
                verts_mv[i, 2] += vertices[k, 2] / 24.0

    L = indices_mv.shape[0]
    M = indices_mv.shape[1]

    # Define new resolution
    L_new = np.unique(indices[:, 0, 0]).size
    M_new = np.unique(indices[0, :, 0]).size
    N_new = np.unique(indices[0, 0, :]).size
    total = L_new * M_new * N_new

    if total < indices.max() + 1:
        raise ValueError(
            "The resolution of the new grid is lower than the maximum value of indices."
        )

    # Calculate center of each cell (with specific resolution)
    verts_rec = np.zeros((total, 3), dtype=float)
    verts_rec_mv = verts_rec

    # Count up for each cell if it is included in the specific resolution
    count_array = np.zeros(total, dtype=np.uint32)
    count_array_mv = count_array

    for i in range(num_cell):
        # Get l, m, n indices from 1D index
        l, m, n = i % L, (i // L) % M, i // (L * M)

        # Get an index from a specific indexing array
        index = indices_mv[l, m, n]

        # Add each center to the reconstructed center
        verts_rec_mv[index, 0] += verts_mv[i, 0]
        verts_rec_mv[index, 1] += verts_mv[i, 1]
        verts_rec_mv[index, 2] += verts_mv[i, 2]

        # Count up for each cell if it is included in the specific resolution
        count_array_mv[index] += 1

    # Divide by the number of cells included in the specific resolution
    for i in range(total):
        verts_rec_mv[i, 0] /= <double>count_array_mv[i]
        verts_rec_mv[i, 1] /= <double>count_array_mv[i]
        verts_rec_mv[i, 2] /= <double>count_array_mv[i]

    return verts_rec.reshape((L_new, M_new, N_new, 3), order="F")
