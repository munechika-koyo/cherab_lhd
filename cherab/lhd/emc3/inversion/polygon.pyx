"""Module for handling polygon's functionalities.

The polygon is defined by the vertices of the EMC3 grid.
"""
cimport cython
cimport numpy as np
from cherab.core.math cimport PolygonMask2D
from raysect.core.math.random cimport uniform
from raysect.core.math.function.float cimport Discrete2DMesh

import numpy as np
import xarray as xr
from scipy.sparse import lil_matrix

from ..grid import Grid
from ..indices import triangulate
from ...tools.fetch import fetch_file

__all__ = ["points_inside_polygon", "generate_boundary_map"]


@cython.initializedcheck(False)
def points_inside_polygon(
    range_l: tuple[int, int],
    range_m: tuple[int, int],
    n: cython.int = -1,
    zone: str = "zone0",
    min_points: cython.int = 200,
    ratio: cython.float = 0.0,
) -> np.ndarray:
    """Generate random points inside the polygon.

    This function is a wrapper of the Cython function
    ``._points_inside_polygon``.

    The polygon is defined by the vertices of the EMC3 grid.

    Parameters
    ----------
    range_l : tuple[int, int]
        Range of radial indices.
    range_m : tuple[int, int]
        Range of poloidal indices.
    n : int
        Index number of toroidal direction, by default -1.
    zone : str
        Name of zone, by default ``"zone0"``.
    min_points : int
        Number of points to generate, by default 200.
    ratio : float
        Ratio to area of the polygon, by default 0.0.
        0.0 means that the number of points is always ``min_points``.

    Returns
    -------
    (N, 2) ndarray
        Array of points inside the polygon.

    Examples
    --------
    >>> points_inside = points_inside_polygon((10, 20), (10, 20), n=0, zone="zone0")
    >>> points_inside.shape
    (134, 2)  # The number of points is random

    Show the points and EMC3 grid at the poloidal cross-section.

    .. code-block:: python

        from matplotlib import pyplot as plt
        from cherab.lhd.emc3 import Grid

        grid = Grid("zone0")
        plt.plot(grid[:, 10, 0, 0], grid[:, 10, 0, 1], "k-")  # plot radial lines (m=10)
        plt.plot(grid[:, 20, 0, 0], grid[:, 20, 0, 1], "k-")  # plot poloidal lines (m=20)
        plt.plot(grid[10, :, 0, 0], grid[10, :, 0, 1], "k-")  # plot poloidal lines (l=10)
        plt.plot(grid[20, :, 0, 0], grid[20, :, 0, 1], "k-")  # plot poloidal lines (l=20)
        plt.plot(points_inside[:, 0], points_inside[:, 1], "r.")  # plot points inside polygon
        plt.xlim(points_inside[:, 0].min() - 0.1, points_inside[:, 0].max() + 0.1)
        plt.ylim((points_inside[:, 1].min() - 0.1, points_inside[:, 1].max() + 0.1))
        plt.xlabel("R (m)"); plt.ylabel("Z (m)")
        plt.show()

    .. image:: ../../_static/images/plotting/points_inside_polygon.png
        :width: 700
        :align: center
        :alt: points_inside_polygon

    """
    cdef:
        np.ndarray points_inside
        double[:, ::1] points_inside_mv

    points_inside_mv = _points_inside_polygon(
        range_l, range_m, n=n, zone=zone, min_points=min_points, ratio=ratio
    )
    points_inside = np.asarray(points_inside_mv)

    return points_inside


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double) _minmax(double[:] arr):
    cdef float min = 1e999
    cdef float max = -1e999
    cdef int i
    for i in range(arr.shape[0]):
        if arr[i] < min:
            min = arr[i]
        if arr[i] > max:
            max = arr[i]
    return min, max


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:, ::1] _points_inside_polygon(
    (int, int) range_l,
    (int, int) range_m,
    int n = -1,
    str zone = "zone0",
    int min_points = 200,
    float ratio = 0.0,
):
    """Generate random points inside the polygon.

    The polygon is defined by the vertices of the EMC3 grid.

    Parameters
    ----------
    range_l : tuple[int, int]
        Range of radial indices.
    range_m : tuple[int, int]
        Range of poloidal indices.
    n : int
        Index number of toroidal direction, by default -1.
    zone : str
        Name of zone, by default ``"zone0"``.
    min_points : int
        Number of minimum points to generate, by default 200.
    ratio : float
        Ratio to area of the polygon, by default 0.0.
        0.0 means that the number of points is always ``min_points``.

    Returns
    -------
    (N, 2) MemoryView
        Array of points inside the polygon.
    """
    # Define local variables
    cdef:
        int l0, l1, m0, m1
        object grid
        double[:, ::1] pol_mv
        int num_points
        PolygonMask2D mask
        double rmin, rmax, zmin, zmax
        double area
        double[:, ::1] points_inside_mv
        double x, y
        bint inside
        int count

    # Unpack range
    l0, l1 = range_l
    m0, m1 = range_m

    # Load EMC3 grid
    grid = Grid(zone)

    if l0 == 0:
        pol_mv = np.vstack(
            (
                grid[l0:l1, m0, n, 0:2],
                grid[l1, m0:m1, n, 0:2],
                grid[l1:l0:-1, m1, n, 0:2],
            )
        )
    else:
        pol_mv = np.vstack(
            (
                grid[l0:l1, m0, n, 0:2],
                grid[l1, m0:m1, n, 0:2],
                grid[l1:l0:-1, m1, n, 0:2],
                grid[l0, m1:m0:-1, n, 0:2],
            )
        )

    mask = PolygonMask2D(pol_mv)

    rmin, rmax = _minmax(pol_mv[:, 0])
    zmin, zmax = _minmax(pol_mv[:, 1])

    # Set number of points based on the polygon area
    area = _polygon_area(pol_mv)
    num_points = max(min_points, <int>(area * ratio))

    points_inside_mv = np.zeros((num_points, 2))
    count = 0
    while count < num_points:
        x = rmin + (rmax - rmin) * uniform()
        y = zmin + (zmax - zmin) * uniform()

        # Check if the point is inside the quadrilateral
        inside = <bint>mask.evaluate(x, y)
        if inside:
            points_inside_mv[count, 0] = x
            points_inside_mv[count, 1] = y
            count += 1

    return points_inside_mv


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _polygon_area(double[:, ::1] vertices) except -1e999:
    """Generate the area of a polygon given its vertices.

    Parameters
    ----------
    vertices : (N, 2) array_like
        An Nx2 array of (x, y) coordinates of the polygon's vertices.

    Returns
    -------
    float
        The area of the polygon
    """
    cdef:
        int n = vertices.shape[0]
        double area = 0.0
        int i, j

    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]

    return 0.5 * abs(area)


def test_polygon_area(vertices):
    return _polygon_area(vertices)


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef object generate_boundary_map(
    str zone1, str zone2, str index_type = "coarse", int min_points = 200, float ratio = 0.0,
):
    """Generate boundary map.

    This function calculates boundary map between two zones.
    In case of a pair of "zone0" and "zone11", which are adjoined each other at the 9 degree in
    toroidal, the map determines which cell of the next (backward) zone the last cell of the
    previous (forward) zone touches.

    .. note::

        Currently, the pairs "zone0" and "zone11" are assumed and verified.
        There is no guarantee that the other pairs will work well.

    Parameters
    ----------
    zone1 : str
        Name of forward zone.
    zone2 : str
        Name of backward zone
    index_type : {"coarse", "cell"}
        Index type, by default ``"coarse"``.
    min_points : int
        Number of minimum points to generate, by default 200.
    ratio : float
        Ratio to area of the polygon, by default 0.0.
        0.0 means that the number of points is always ``min_points``.

    Returns
    -------
    (N, M) scipy.sparse.lil_matrix
        Array of boundary map between zone1 and zone2.

    Examples
    --------
    >>> vmap = generate_boundary_map("zone0", "zone11", index_type="coarse")
    """
    cdef:
        long[::1] indices_radial_mv, indices_poloidal_mv
        np.uint32_t[:, :, ::1] indices_mv
        Discrete2DMesh mesh
        int index_size
        object boundary_map, grid, verts, tris, data
        int num_radial_index
        int m, l, i, index
        double[:, ::1] points_inside_mv
        int num_points

    # validation
    if min_points <= 0:
        raise ValueError("Number of points must be positive")

    # load index data
    path = fetch_file("emc3/grid-360.nc")
    with xr.open_dataset(path, group=f"/{zone1}/index")[index_type] as da:
        indices_radial_mv = da.attrs["indices_radial"]
        indices_poloidal_mv = da.attrs["indices_poloidal"]
        indices_mv = da.data

    # create 2D mesh
    grid = Grid(zone2)
    verts, tris, data = triangulate(grid, index_type=index_type, n_phi=0)
    index_size = max(data) + 1
    mesh = Discrete2DMesh(verts, tris, data, limit=False, default_value=-1)

    # create boundary map sparse matrix
    boundary_map = lil_matrix(
        (indices_mv[indices_mv.shape[0] - 1, indices_mv.shape[1] - 1, 0] + 1, index_size),
        dtype=np.float64
    )
    num_radial_index = indices_radial_mv.shape[0] - 1

    for m in range(0, indices_poloidal_mv.shape[0] - 1):
        for l in range(0, indices_radial_mv.shape[0] - 1):

            # generate points inside the polygon
            points_inside_mv = _points_inside_polygon(
                (indices_radial_mv[l], indices_radial_mv[l + 1]),
                (indices_poloidal_mv[m], indices_poloidal_mv[m + 1]),
                n=-1,
                min_points=min_points,
                ratio=ratio,
                zone=zone1,
            )

            # find the corresponding mesh index for each point
            num_points = points_inside_mv.shape[0]
            for i in range(num_points):
                index = <int>mesh.evaluate(points_inside_mv[i, 0], points_inside_mv[i, 1])
                if index > -1:
                    boundary_map[l + m * num_radial_index, index] += 1

            # normalize
            boundary_map[l + m * num_radial_index, :] /= <double>num_points

    return boundary_map
