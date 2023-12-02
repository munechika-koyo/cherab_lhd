"""Module to deal with EMC3-EIRENE-defined grids."""
cimport numpy as np
cimport cython
from libc.math cimport cos, sin, M_PI
from raysect.primitive.mesh cimport TetraMesh

import warnings
from pathlib import Path
from types import EllipsisType

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

from .cython.tetrahedralization cimport tetrahedralize

from ..machine.wall import adjacent_toroidal_angles, periodic_toroidal_angle
from ..tools.spinner import Spinner
from .repository.utility import DEFAULT_HDF5_PATH, DEFAULT_TETRA_MESH_PATH

__all__ = ["Grid", "plot_grids_rz", "install_tetra_meshes"]

cdef list ZONES = [
    ["zone0", "zone1", "zone2", "zone3", "zone4"],  # zone_type = 1
    ["zone11", "zone12", "zone13", "zone14", "zone15"],  # zone_type 2
]
# Const.
cdef:
    double RMIN = 2.0  # [m]
    double RMAX = 5.5
    double ZMIN = -1.6
    double ZMAX = 1.6

# Plotting config.
cdef dict LINE_STYLE = {"color": "black", "linewidth": 0.5}


cdef class Grid:
    """Class for dealing with grid coordinates defined by EMC3-EIRENE.

    This class handles originally defined EMC3-EIRENE grid coordinates in :math:`(R, Z, \\varphi)`,
    and offers methods to produce cell vertices in :math:`(X, Y, Z)` coordinates and
    their indices, which a **cell** means a cubic-like mesh with 8 vertices.
    Using these data, procedure of generating a :obj:`~raysect.primitive.mesh.tetra_mesh.TetraMesh`
    instance is also implemented.

    | Total number of grids coordinates is L x M x N, each letter of which means:
    | L: Radial grid resolution
    | M: Poloidal grid resolution
    | N: Toroidal grid resolution.


    Parameters
    ----------
    zone
        name of grid zone. Users can select only one option of ``"zone0"`` - ``"zone21"``.
    grid_group
        name of grid group corresponding to magnetic axis configuration, by default ``grid-360``.
    hdf5_path
        path to the HDF5 file storing grid dataset, by default ``~/.cherab/lhd/emc3.hdf5``.


    Examples
    --------
    .. prompt:: python >>> auto

        >>> grid = Grid("zone0")
        >>> grid
        Grid(zone='zone0', grid_group='grid-360')
        >>> str(grid)
        'Grid for (zone: zone0, L: 82, M: 601, N: 37, number of cells: 1749600)'
    """

    def __init__(
        self, zone: str, grid_group: str = "grid-360", hdf5_path: Path | str = DEFAULT_HDF5_PATH
    ) -> None:
        cdef:
            object file
            object dset

        # === Parameters validation ================================================================
        # set and validate hdf5_path
        if isinstance(hdf5_path, (Path, str)):
            self._hdf5_path = Path(hdf5_path)
        else:
            raise TypeError("hdf5_path must be a string or a pathlib.Path instance.")
        if not self._hdf5_path.exists():
            raise FileNotFoundError(f"{self._hdf5_path.name} file does not exist.")

        # set properties
        self._zone = zone
        self._grid_group = grid_group

        # === Load grid data from HDF5 file
        with h5py.File(self._hdf5_path, mode="r") as file:
            # load grid dataset
            dset = file[grid_group][zone]["grids"]

            # Load grid configuration
            self._config = GridConfig(
                L=dset.attrs["L"],
                M=dset.attrs["M"],
                N=dset.attrs["N"],
                num_cells=dset.attrs["num_cells"],
            )

            # Load grid coordinates data
            self._grid_data = dset[:]

        # set shape
        self._shape = self._config.L, self._config.M, self._config.N

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zone={self.zone!r}, grid_group={self.grid_group!r})"

    def __str__(self) -> str:
        L, M, N, num_cells = (
            self._config.L,
            self._config.M,
            self._config.N,
            self._config.num_cells
        )
        return f"{self.__class__.__name__} for (zone: {self.zone}, L: {L}, M: {M}, N: {N}, number of cells: {num_cells})"

    def __getitem__(
        self, key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | NDArray
    ) -> NDArray[np.float64] | float:
        """Return grid coordinates indexed by (l, m, n, rzphi).

        Returned grid coordinates are in :math:`(R, Z, \\varphi)` which can be specified by
        ``l``: radial, ``m``: poloidal, ``n``: torodial indices.

        Examples
        --------
        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid[0, 0, 0, :]  # (l=0, m=0, n=0)
            array([3.593351e+00, 0.000000e+00, 0.000000e+00])  # (R, Z, phi) coordinates

            >>> grid[:, -10, 0, :]  # (radial coords at m=-10, n=0)
            array([[3.600000e+00, 0.000000e+00, 0.000000e+00],
                   [3.593494e+00, 3.076000e-03, 0.000000e+00],
                   [3.560321e+00, 1.875900e-02, 0.000000e+00],
                   ...,
                   [3.267114e+00, 1.573770e-01, 0.000000e+00]])
        """
        return self._grid_data[key]

    @property
    def hdf5_path(self) -> Path:
        """:obj:`~pathlib.Path`: HDF5 dataset file path."""
        return self._hdf5_path

    @property
    def zone(self) -> str:
        """str: Name of zone."""
        return self._zone

    @property
    def grid_group(self) -> str:
        """str: Name of grid group."""
        return self._grid_group

    @property
    def shape(self) -> tuple[int, int, int]:
        """tuple[int, int, int]: Shape of grid (L, M, N)."""
        return self._shape

    @property
    def config(self) -> dict[str, int]:
        """dict[str, int]: Configuration dictionary containing grid resolutions and number of cells.

        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.config
            {'L': 82, 'M': 601, 'N': 37, 'num_cells': 1749600}
        """
        return {
            "L": self._config.L,
            "M": self._config.M,
            "N": self._config.N,
            "num_cells": self._config.num_cells
        }

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """numpy.ndarray: Raw Grid coordinates data array.

        The dimension of array is 4 dimension, shaping ``(L, M, N, 3)``.
        The coordinate is :math:`(R, Z, \\phi)`. :math:`\\phi` is in [degree].

        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.grid_data.shape
            (82, 601, 37, 3)
            >>> grid.grid_data
            array([[[[ 3.600000e+00,  0.000000e+00,  0.000000e+00],
                     [ 3.600000e+00,  0.000000e+00,  2.500000e-01],
                     [ 3.600000e+00,  0.000000e+00,  5.000000e-01],
                    ...
                     [ 3.096423e+00, -7.012100e-02,  8.500000e+00],
                     [ 3.087519e+00, -6.796400e-02,  8.750000e+00],
                     [ 3.078508e+00, -6.543900e-02,  9.000000e+00]]]])
        """
        return self._grid_data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray generate_vertices(self):
        """Generate grid vertices array. A `grid_data` array is converted to 2D
        array which represents a vertex in :math:`(X, Y, Z)` coordinates.

        The vertices array is stacked in the order of `(L, M, N)`.

        Returns
        -------
        :obj:`~numpy.ndarray`
            vertices (L x N x M, 3) 2D array


        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> verts = grid.generate_vertices()
            >>> verts.shape
            (1823434, 3)
            >>> verts
            array([[ 3.6       ,  0.        ,  0.        ],
                   [ 3.593351  ,  0.        , -0.        ],
                   [ 3.559212  ,  0.        , -0.        ],
                   ...,
                   [ 3.04105882,  0.4816564 , -0.065465  ],
                   [ 3.04083165,  0.48162042, -0.065452  ],
                   [ 3.04060646,  0.48158475, -0.065439  ]])
        """
        cdef:
            int L, M, N
            np.ndarray[np.float64_t, ndim=4] vertices
            np.float64_t[:, :, :, ::1] grid_mv
            float phi

        L, M, N = self._shape
        vertices = np.zeros((L, M, N, 3), dtype=np.float64)
        grid_mv = self._grid_data

        for n in range(N):
            phi = grid_mv[0, 0, n, 2] * M_PI / 180.0
            vertices[:, :, n, 0] = self._grid_data[:, :, n, 0] * cos(phi)
            vertices[:, :, n, 1] = self._grid_data[:, :, n, 0] * sin(phi)
            vertices[:, :, n, 2] = self._grid_data[:, :, n, 1]

        return vertices.reshape((L * M * N, 3), order="F")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray generate_cell_indices(self):
        """Generate cell indices array.

        One row of cell indices array represents one cubic-like mesh with 8 vertices.
        Cells are indexed in the order of `(L, M, N)` direction.

        Returns
        -------
        :obj:`~numpy.ndarray`
            cell indices ((L-1) x (M-1) x (N-1), 8) 2D array


        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> cells = grid.generate_cell_indices()
            >>> cells.shape
            (1749600, 8)
            >>> cells
            array([[      0,       1,      83, ...,   49283,   49365,   49364],
                   [      1,       2,      84, ...,   49284,   49366,   49365],
                   [      2,       3,      85, ...,   49285,   49367,   49366],
                   ...,
                   [1774066, 1774067, 1774149, ..., 1823349, 1823431, 1823430],
                   [1774067, 1774068, 1774150, ..., 1823350, 1823432, 1823431],
                   [1774068, 1774069, 1774151, ..., 1823351, 1823433, 1823432]],
                  dtype=uint32)
        """
        cdef:
            int L, M, N
            np.ndarray[np.uint32_t, ndim=2] cells
            np.uint32_t[:, ::1] cells_mv
            int i = 0
            int l, m, n

        L, M, N = self._shape
        cells = np.zeros(((L - 1) * (M - 1) * (N - 1), 8), dtype=np.uint32)
        cells_mv = cells

        for n in range(0, N - 1):
            for m in range(0, M - 1):
                for l in range(0, L - 1):
                    cells_mv[i, 0] = M * L * n + L * m + l
                    cells_mv[i, 1] = M * L * n + L * m + l + 1
                    cells_mv[i, 2] = M * L * n + L * (m + 1) + l + 1
                    cells_mv[i, 3] = M * L * n + L * (m + 1) + l
                    cells_mv[i, 4] = M * L * (n + 1) + L * m + l
                    cells_mv[i, 5] = M * L * (n + 1) + L * m + l + 1
                    cells_mv[i, 6] = M * L * (n + 1) + L * (m + 1) + l + 1
                    cells_mv[i, 7] = M * L * (n + 1) + L * (m + 1) + l
                    i += 1
        return cells

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        n_phi: int = 0,
        rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plotting EMC3-EIRENE-defined grids in :math:`r - z` plane.

        Parameters
        ----------
        fig
            matplotlib figure object, by default ``fig = plt.figure(dpi=200)``
        ax
            matplotlib axes object, by default ``ax = fig.add_subplot()``.
        n_phi
            index of toroidal grid, by default 0
        rz_range
            sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
            by default ``(2.0, 5.5, -1.6, 1.6)``
        **kwargs: :obj:`matplotlib.lines.Line2D` properties, optional
            *kwargs* are used to specify properties,
            by default ``{"color": "black", "linewidth": 0.5}``

        Returns
        -------
            tuple of matplotlib figure and axes object


        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.plot()

        .. image:: ../../_static/images/plotting/grid_zone0.png
        """
        cdef:
            float rmin, rmax, zmin, zmax
            int L, M, l, m

        rmin, rmax, zmin, zmax = rz_range
        if rmin >= rmax or zmin >= zmax:
            raise ValueError("Invalid rz_range")

        # set default line style
        for key, value in LINE_STYLE.items():
            kwargs.setdefault(key, value)

        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots(dpi=200)
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        L, M, _ = self._shape

        # plot radial line
        if self.zone in {"zone0", "zone11"}:
            num_pol = M - 1
        else:
            num_pol = M
        for m in range(num_pol):
            ax.plot(self._grid_data[:, m, n_phi, 0], self._grid_data[:, m, n_phi, 1], **kwargs)
        # plot poloidal line
        for l in range(L):
            ax.plot(self._grid_data[l, :, n_phi, 0], self._grid_data[l, :, n_phi, 1], **kwargs)

        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)

        ax.text(
            rmin + (rmax - rmin) * 0.02,
            zmax - (zmax - zmin) * 0.02,
            f"$\\phi=${self.grid_data[0, 0, n_phi, 2]:.2f}$^\\circ$",
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
        )
        set_axis_properties(ax)

        return (fig, ax)

    def plot_coarse(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        n_phi: int = 0,
        rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plotting EMC-EIRENE-defined coarse grids in :math:`r - z` plane.

        The indices to use as the coarse grid is stored in attributes of `"/index/coarse"` HDF5
        group. So this method is available only if they are stored.
        Creating coarse grid indices is achieved by executing
        :obj:`~cherab.lhd.emc3.indices.create_new_index`.


        Parameters
        ----------
        fig
            matplotlib figure object, by default ``fig = plt.figure(dpi=200)``
        ax
            matplotlib axes object, by default ``ax = fig.add_subplot()``.
        n_phi
            index of toroidal grid, by default 0
        rz_range
            sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
            by default ``(2.0, 5.5, -1.6, 1.6)``
        **kwargs: :obj:`matplotlib.lines.Line2D` properties, optional
            *kwargs* are used to specify properties,
            by default ``{"color": "black", "linewidth": 0.5}``

        Returns
        -------
            tuple of matplotlib figure and axes object


        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.plot_coarse()

        .. image:: ../../_static/images/plotting/grid_coarse_zone0.png
        """
        cdef:
            float rmin, rmax, zmin, zmax
            int l, m
            np.ndarray radial_indices
            np.ndarray poloidal_indices

        rmin, rmax, zmin, zmax = rz_range
        if rmin >= rmax or zmin >= zmax:
            raise ValueError("Invalid rz_range")

        # set default line style
        for key, value in LINE_STYLE.items():
            kwargs.setdefault(key, value)

        # load coarse grid indices
        with h5py.File(DEFAULT_HDF5_PATH, mode="r") as file:
            try:
                ds = file["grid-360"][self.zone]["index"]["coarse"]
                radial_indices = ds.attrs["radial indices"]
                poloidal_indices = ds.attrs["poloidal indices"]

            except Exception as err:
                raise ValueError("Cannot load coarse grid attributes") from err

        # === plotting =============================================================================
        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots(dpi=200)
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        if self.zone in {"zone0", "zone11"}:
            poloidal_indices = poloidal_indices[:-1]

        # plot radial line
        for m in poloidal_indices:
            ax.plot(self._grid_data[:, m, n_phi, 0], self._grid_data[:, m, n_phi, 1], **kwargs)
        # plot poloidal line
        for l in radial_indices:
            ax.plot(self._grid_data[l, :, n_phi, 0], self._grid_data[l, :, n_phi, 1], **kwargs)

        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)

        ax.text(
            rmin + (rmax - rmin) * 0.02,
            zmax - (zmax - zmin) * 0.02,
            f"$\\phi=${self._grid_data[0, 0, n_phi, 2]:.2f}$^\\circ$",
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
        )
        set_axis_properties(ax)

        return (fig, ax)

    def plot_outline(
        self,
        phi: float = 0.0,
        fig: Figure | None = None,
        ax: Axes | None = None,
        show_phi: bool = True,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plotting EMC3-EIRENE-defined grid outline in :math:`r - z` plane.

        This method allows users to plot grid outline at a specific toroidal angle :math:`\\varphi`.
        The toroidal angle is arbitrary, where the grid outline is calculated by linear interpolation
        between two nearest toroidal grids.

        Parameters
        ----------
        phi
            toroidal grid in [degree], by default 0.0
        fig
            matplotlib figure object, by default ``fig = plt.figure(dpi=200)``
        ax
            matplotlib axes object, by default ``ax = fig.add_subplot()``.
        show_phi
            show toroidal angle text in the plot, by default True
        **kwargs: :obj:`matplotlib.lines.Line2D` properties, optional
            *kwargs* are used to specify properties,
            by default ``{"color": "black", "linewidth": 0.5}``

        Returns
        -------
            tuple of matplotlib figure and axes object


        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.plot_outline(4.2)

        .. image:: ../../_static/images/plotting/grid_outline_zone0.png
        """
        # === generate interpolated grids ==========================================================
        # put phi in [0, 18) range
        phi_t, fliped = periodic_toroidal_angle(phi)

        phi_range = self._grid_data[0, 0, 0, 2], self._grid_data[0, 0, -1, 2]
        if phi_t < phi_range[0] or phi_t > phi_range[1]:
            raise ValueError(f"toroidal angle {phi_t} is out of grid range {phi_range}.")

        # find adjacent phis
        phi_left_index, phi_right_index = adjacent_toroidal_angles(
            phi_t, self._grid_data[0, 0, :, 2]
        )

        # define phi_left, phi_right
        phi_left = self._grid_data[0, 0, phi_left_index, 2]
        phi_right = self._grid_data[0, 0, phi_right_index, 2]

        # load rz grids at adjacent phis
        grid_left = self._grid_data[:, :, phi_left_index, :2].copy()
        grid_right = self._grid_data[:, :, phi_right_index, :2].copy()

        # fliped value for z axis
        if fliped:
            grid_left[:, :, 1] *= -1
            grid_right[:, :, 1] *= -1

        # linearly interpolate grid
        grid = ((phi_t - phi_left) * grid_right + (phi_right - phi_t) * grid_left) / (
            phi_right - phi_left
        )

        # === plotting =============================================================================
        # set default line style
        for key, value in LINE_STYLE.items():
            kwargs.setdefault(key, value)

        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots(dpi=200)
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        # plot outline (last poloidal line)
        ax.plot(grid[-1, :, 0], grid[-1, :, 1], **kwargs)

        if self.zone not in {"zone0", "zone11"}:
            # plot first poloidal line
            ax.plot(grid[0, :, 0], grid[0, :, 1], **kwargs)

            # plot first/last radial lines
            ax.plot(grid[:, 0, 0], grid[:, 0, 1], **kwargs)
            ax.plot(grid[:, -1, 0], grid[:, -1, 1], **kwargs)

        if show_phi:
            rmin, rmax = ax.get_xlim()
            zmin, zmax = ax.get_ylim()

            ax.text(
                rmin + (rmax - rmin) * 0.02,
                zmax - (zmax - zmin) * 0.02,
                f"$\\phi=${phi:.2f}$^\\circ$",
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
            )
        set_axis_properties(ax)

        return (fig, ax)


def plot_grids_rz(
    fig: Figure | None = None,
    ax: Axes | None = None,
    zone_type: int = 1,
    n_phi: int = 0,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plotting EMC-EIRENE-defined grids in :math:`r - z` plane.

    Parameters
    ----------
    fig
        matplotlib figure object, by default ``plt.figure(dpi=200)``
    ax
        matplotlib axes object, by default ``ax = fig.add_subplot()``.
    zone_type
        type of zones collections, by default 1

        | type 1 is ``["zone0", "zone1", "zone2", "zone3", "zone4"]``,
        | type 2 is ``["zone11", "zone12", "zone13", "zone14", "zone15"]``.
    n_phi
        index of toroidal grid, by default 0
    rz_range
        sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``
    **kwargs: :obj:`matplotlib.lines.Line2D` properties, optional
        *kwargs* are used to specify properties,
        by default ``{"color": "black", "linewidth": 0.5}``

    Returns
    -------
        tuple of matplotlib figure and axes objects

    Examples
    --------
    .. prompt:: python >>> auto

        >>> plot_grids_rz(zone_type=1, n_phi=10)

    .. image:: ../../_static/images/plotting/plot_grids_rz.png
    """
    cdef:
        float rmin, rmax, zmin, zmax
        int L, M, l, m

    # validate parameters
    if zone_type not in {1, 2}:
        raise ValueError(f"zone_type must be either 1 or 2. (zone_type: {zone_type})")
    zone_type -= 1

    rmin, rmax, zmin, zmax = rz_range
    if rmin >= rmax or zmin >= zmax:
        raise ValueError("Invalid rz_range")

    # set default line style
    for key, value in LINE_STYLE.items():
        kwargs.setdefault(key, value)

    # create figure/axes object
    if not isinstance(ax, Axes):
        if not isinstance(fig, Figure):
            fig, ax = plt.subplots(dpi=200)
        else:
            ax = fig.add_subplot()

    ax.set_aspect("equal")

    for zone in ZONES[zone_type]:
        emc = Grid(zone=zone)
        L, M, _ = emc.shape

        # plot radial line
        if zone in {"zone0", "zone11"}:
            num_pol = M - 1
        else:
            num_pol = M
        for m in range(num_pol):
            ax.plot(emc.grid_data[:, m, n_phi, 0], emc.grid_data[:, m, n_phi, 1], **kwargs)
        # plot poloidal line
        for l in range(L):
            ax.plot(emc.grid_data[l, :, n_phi, 0], emc.grid_data[l, :, n_phi, 1], **kwargs)

    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)

    ax.text(
        rmin + (rmax - rmin) * 0.02,
        zmax - (zmax - zmin) * 0.02,
        f"$\\phi=${emc.grid_data[0, 0, n_phi, 2]:.2f}$^\\circ$",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
    )
    set_axis_properties(ax)

    return (fig, ax)


def plot_grids_coarse(
    fig: Figure | None = None,
    ax: Axes | None = None,
    zone_type: int = 1,
    n_phi: int = 0,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plotting EMC-EIRENE-defined coarse grids in :math:`r - z` plane.

    The indices to use as the coarse grid is stored in attributes of `"/index/coarse"` HDF5 group.

    Parameters
    ----------
    fig
        matplotlib figure object, by default ``plt.figure(dpi=200)``
    ax
        matplotlib axes object, by default ``ax = fig.add_subplot()``.
    zone_type
        type of zones collections, by default 1

        | type 1 is ``["zone0", "zone1", "zone2", "zone3", "zone4"]``,
        | type 2 is ``["zone11", "zone12", "zone13", "zone14", "zone15"]``.
    n_phi
        index of toroidal grid, by default 0
    rz_range
        sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``
    **kwargs: :obj:`matplotlib.lines.Line2D` properties, optional
        *kwargs* are used to specify properties,
        by default ``{"color": "black", "linewidth": 0.5}``

    Returns
    -------
        tuple of matplotlib figure and axes objects

    Examples
    --------
    .. prompt:: python >>> auto

        >>> plot_grids_coarse(zone_type=1, n_phi=10)

    .. image:: ../../_static/images/plotting/plot_grids_coarse.png
    """
    # validate parameters
    if zone_type not in {1, 2}:
        raise ValueError(f"zone_type must be either 1 or 2. (zone_type: {zone_type})")
    zone_type -= 1

    rmin, rmax, zmin, zmax = rz_range
    if rmin >= rmax or zmin >= zmax:
        raise ValueError("Invalid rz_range")

    # set default line style
    for key, value in LINE_STYLE.items():
        kwargs.setdefault(key, value)

    if not isinstance(ax, Axes):
        if not isinstance(fig, Figure):
            fig, ax = plt.subplots(dpi=200)
        else:
            ax = fig.add_subplot()

    ax.set_aspect("equal")
    for zone in ZONES[zone_type]:
        # load coarse grid indices
        with h5py.File(DEFAULT_HDF5_PATH, mode="r+") as file:
            try:
                ds = file["grid-360"][zone]["index"]["coarse"]
                radial_indices: np.ndarray = ds.attrs["radial indices"]
                poloidal_indices: np.ndarray = ds.attrs["poloidal indices"]

            except Exception:
                warnings.warn(f"Cannot load coarse grid attributes in {zone}.", stacklevel=2)
                continue

        emc = Grid(zone=zone)

        # plot radial line
        if zone in {"zone0", "zone11"}:
            poloidal_indices = poloidal_indices[:-1]

        # plot radial line
        for m in poloidal_indices:
            ax.plot(emc.grid_data[:, m, n_phi, 0], emc.grid_data[:, m, n_phi, 1], **kwargs)
        # plot poloidal line
        for l in radial_indices:
            ax.plot(emc.grid_data[l, :, n_phi, 0], emc.grid_data[l, :, n_phi, 1], **kwargs)

    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)

    ax.text(
        rmin + (rmax - rmin) * 0.02,
        zmax - (zmax - zmin) * 0.02,
        f"$\\phi=${emc.grid_data[0, 0, n_phi, 2]:.2f}$^\\circ$",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
    )
    set_axis_properties(ax)

    return (fig, ax)


def set_axis_properties(axes: Axes):
    """Set x-, y-axis property. This function set axis labels and tickers.

    Parameters
    ----------
    axes
        matplotlib Axes object
    """
    axes.set_xlabel("$R$ [m]")
    axes.set_ylabel("$Z$ [m]")
    axes.xaxis.set_minor_locator(MultipleLocator(0.1))
    axes.yaxis.set_minor_locator(MultipleLocator(0.1))
    axes.xaxis.set_major_formatter("{x:.1f}")
    axes.yaxis.set_major_formatter("{x:.1f}")
    axes.tick_params(direction="in", labelsize=10, which="both", top=True, right=True)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void install_tetra_meshes(
    zones: list[str] = ZONES[0] + ZONES[1],
    tetra_mesh_path: Path | str = DEFAULT_TETRA_MESH_PATH,
    update: bool = True,
    grid_group: str = "grid-360",
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
):
    """Create :obj:`~raysect.primitive.mesh.tetra_mesh.TetraMesh` .rsm files
    and install them into a repository.

    Default repository is set to ``~/.cherab/lhd/tetra/``, and automatically created if it does not
    exist. The file name is determined by each zone name like ``zone0.rsm``.

    .. note::

        It takes a lot of time to calculate all TetraMesh instance because each zone has numerous
        number of grids.

    Parameters
    ----------
    zones
        list of zone names, by default ``["zone0",..., "zone4", "zone11",..., "zone15"]``
    tetra_mesh_path
        path to the directory to save TetraMesh .rsm files, by default ``~/.cherab/lhd/tetra/``
    update
        whether or not to update existing TetraMesh .rsm file, by default True
    **kwargs : :obj:`.Grid` properties, optional
        *kwargs* are used to specify :obj:`.Grid` properties except for ``zone`` argument.
    """
    cdef:
        str zone
        object tetra_path
        Grid emc
        np.ndarray[np.float64_t] vertices
        np.ndarray[np.uint32_t] tetrahedra
        TetraMesh tetra

    if isinstance(tetra_mesh_path, (Path, str)):
        tetra_mesh_path = Path(tetra_mesh_path)
    else:
        raise TypeError("tetra_mesh_path must be a string or pathlib.Path instance.")

    # make directory
    tetra_mesh_path.mkdir(parents=True, exist_ok=True)

    # populate each zone TetraMesh instance
    for zone in zones:
        # path to the tetra .rsm file
        tetra_path = tetra_mesh_path / f"{zone}.rsm"

        with Spinner(text=f"Constructing {zone} tetra mesh...", timer=True) as sp:
            # skip if it exists and update is False
            if tetra_path.exists() and not update:
                sp.ok("⏩")
                continue

            # Load EMC3 grid
            emc = Grid(zone, grid_group=grid_group, hdf5_path=hdf5_path)

            # prepare vertices and tetrahedral indices
            vertices = emc.generate_vertices()
            tetrahedra = tetrahedralize(emc.generate_cell_indices())

            # create TetraMesh instance (heavy calculation)
            tetra = TetraMesh(vertices, tetrahedra, tolerant=False)

            # save
            tetra.save(tetra_path)

            sp.ok()
