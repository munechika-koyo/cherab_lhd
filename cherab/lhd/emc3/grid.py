"""Module to deal with EMC3-EIRENE-defined grids."""
from pathlib import Path
from types import EllipsisType
from typing import Any

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from raysect.primitive.mesh import TetraMesh

from ..machine.wall import adjacent_toroidal_angles, periodic_toroidal_angle
from ..tools.spinner import Spinner
from ..tools.visualization import set_axis_properties
from .cython import tetrahedralize
from .repository.utility import DEFAULT_HDF5_PATH, DEFAULT_TETRA_MESH_PATH

__all__ = ["Grid", "plot_grids_rz", "install_tetra_meshes"]

ZONES = [
    ["zone0", "zone1", "zone2", "zone3", "zone4"],  # zone_type = 1
    ["zone11", "zone12", "zone13", "zone14", "zone15"],  # zone_type 2
]
# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6

# Plotting config.
LINE_STYLE = {"color": "black", "linewidth": 0.25}


class Grid:
    """Class for dealing with grid coordinates defined by EMC3-EIRENE.

    This class handles originally defined EMC3-EIRENE grid coordinates in :math:`(R, Z, \\varphi)`,
    and offers methods to produce cell vertices in :math:`(X, Y, Z)` coordinates and
    their indices, which a ``cell'' means a cubic-like mesh with 8 vertices.
    Using these data, procedure of generating a :obj:`~raysect.primitive.mesh.tetra_mesh.TetraMesh`
    instance is also implemented.

    | Total number of grids coordinates is L x M x N, each letter of which means:
    | L: Radial grid resolution
    | M: Poloidal grid resolution
    | N: Toroidal grid resolution.


    Parameters
    ----------
    zone
        name of grid zone. Users can select only one option of ``"zone0" - "zone21"``.
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
        with h5py.File(self._hdf5_path, mode="r") as h5_file:
            # load grid dataset
            dset = h5_file[grid_group][zone]["grids"]

            # Load grid configuration
            self._grid_config = dict(
                L=dset.attrs["L"],
                M=dset.attrs["M"],
                N=dset.attrs["N"],
                num_cells=dset.attrs["num_cells"],
            )

            # Load grid coordinates data
            self._grid_data = dset[:]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zone={self.zone!r}, grid_group={self.grid_group!r})"

    def __str__(self) -> str:
        L, M, N, num_cells = self._grid_config.values()
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
        return self.grid_data[key]

    @property
    def hdf5_path(self) -> Path:
        """HDF5 dataset file path."""
        return self._hdf5_path

    @property
    def zone(self) -> str:
        """Name of zone."""
        return self._zone

    @property
    def grid_group(self) -> str:
        """Name of grid group."""
        return self._grid_group

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of grid (L, M, N)."""
        return self.grid_config["L"], self.grid_config["M"], self.grid_config["N"]

    @property
    def grid_config(self) -> dict[str, int]:
        """Configuration dictionary containing grid resolutions and number of cells.

        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.grid_config
            {'L': 82, 'M': 601, 'N': 37, 'num_cells': 1749600}
        """
        return self._grid_config

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """Raw Grid coordinates data array.

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

    def generate_vertices(self) -> NDArray[np.float64]:
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
        L, M, N = self.shape
        vertices = np.zeros_like(self._grid_data)
        grid = self._grid_data

        for n in range(N):
            phi = np.deg2rad(grid[0, 0, n, 2])
            vertices[:, :, n, 0] = grid[:, :, n, 0] * np.cos(phi)
            vertices[:, :, n, 1] = grid[:, :, n, 0] * np.sin(phi)
            vertices[:, :, n, 2] = grid[:, :, n, 1]

        return vertices.reshape((L * M * N, 3), order="F")

    def generate_cell_indices(self) -> NDArray[np.uint32]:
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
        L, M, N = self.shape
        cells = np.zeros(((L - 1) * (M - 1) * (N - 1), 8), dtype=np.uint32)
        i = 0
        for n in range(0, N - 1):
            for m in range(0, M - 1):
                for l in range(0, L - 1):
                    cells[i, :] = (
                        M * L * n + L * m + l,
                        M * L * n + L * m + l + 1,
                        M * L * n + L * (m + 1) + l + 1,
                        M * L * n + L * (m + 1) + l,
                        M * L * (n + 1) + L * m + l,
                        M * L * (n + 1) + L * m + l + 1,
                        M * L * (n + 1) + L * (m + 1) + l + 1,
                        M * L * (n + 1) + L * (m + 1) + l,
                    )
                    i += 1
        return cells

    def plot(
        self,
        fig: Figure | None = None,
        ax: Axes | None = None,
        n_phi: int = 0,
        rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
        linestyle: dict[str, Any] = LINE_STYLE,
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
        linestyle
            line style, by default ``{"color": "black", "linewidth": 0.25}``

        Returns
        -------
            tuple of matplotlib figure and axes object


        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.plot()

        .. image:: ../_static/images/plotting/grid_zone0.png
        """
        rmin, rmax, zmin, zmax = rz_range
        if rmin >= rmax or zmin >= zmax:
            raise ValueError("Invalid rz_range")

        # set default line style
        if not isinstance(linestyle, dict):
            raise TypeError("linestyle must be a dict")

        linestyle.setdefault("color", "black")
        linestyle.setdefault("linewidth", 0.5)

        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots(dpi=200)
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        L, M, _ = self.shape

        # plot radial line
        if self.zone in {"zone0", "zone11"}:
            num_pol = M - 1
        else:
            num_pol = M
        for m in range(num_pol):
            ax.plot(
                self.grid_data[:, m, n_phi, 0],
                self.grid_data[:, m, n_phi, 1],
                **linestyle
            )
        # plot poloidal line
        for l in range(L):
            ax.plot(
                self.grid_data[l, :, n_phi, 0],
                self.grid_data[l, :, n_phi, 1],
                **linestyle
            )

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
        ax.set_xlabel("R[m]")
        ax.set_ylabel("Z[m]")

        return (fig, ax)

    def plot_outline(
        self,
        phi: float = 0.0,
        fig: Figure | None = None,
        ax: Axes | None = None,
        linestyle: dict[str, Any] = LINE_STYLE,
        show_phi: bool = True,
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
        linestyle
            line style, by default ``{"color": "black", "linewidth": 0.25}``
        show_phi
            show toroidal angle text in the plot, by default True

        Returns
        -------
            tuple of matplotlib figure and axes object


        .. prompt:: python >>> auto

            >>> grid = Grid("zone0")
            >>> grid.plot_outline(4.2)

        .. image:: ../_static/images/plotting/grid_zone0_outline.png
        """
        # === Parameters validation ================================================================
        if not isinstance(linestyle, dict):
            raise TypeError("linestyle must be a dict")

        # === generate interpolated grids ==========================================================
        # put phi in [0, 18) range
        phi_t, fliped = periodic_toroidal_angle(phi)

        phi_range = self.grid_data[0, 0, 0, 2], self.grid_data[0, 0, -1, 2]
        if phi_t < phi_range[0] or phi_t > phi_range[1]:
            raise ValueError(f"toroidal angle {phi_t} is out of grid range {phi_range}.")

        # find adjacent phis
        phi_left_index, phi_right_index = adjacent_toroidal_angles(
            phi_t, self.grid_data[0, 0, :, 2]
        )

        # load rz grids at adjacent phis
        grid_left = self.grid_data[:, :, phi_left_index, :2]
        grid_right = self.grid_data[:, :, phi_right_index, :2]

        # fliped value for z axis
        if fliped:
            grid_left[:, :, 1] *= -1
            grid_right[:, :, 1] *= -1

        # linearly interpolate grid
        phi_left = self.grid_data[0, 0, phi_left_index, 2]
        phi_right = self.grid_data[0, 0, phi_right_index, 2]
        grid = ((phi_t - phi_left) * grid_right + (phi_right - phi_t) * grid_left) / (
            phi_right - phi_left
        )

        # === plotting =============================================================================
        # set default line style
        linestyle.setdefault("color", "black")
        linestyle.setdefault("linewidth", 0.5)

        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots(dpi=200)
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        # plot outline (last poloidal line)
        ax.plot(grid[-1, :, 0], grid[-1, :, 1], **linestyle)

        if self.zone not in {"zone0", "zone11"}:
            # plot first poloidal line
            ax.plot(grid[0, :, 0], grid[0, :, 1], **linestyle)

            # plot first/last radial lines
            ax.plot(grid[:, 0, 0], grid[:, 0, 1], **linestyle)
            ax.plot(grid[:, -1, 0], grid[:, -1, 1], **linestyle)


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
        ax.set_xlabel("R[m]")
        ax.set_ylabel("Z[m]")

        return (fig, ax)


def plot_grids_rz(
    fig: Figure | None = None,
    ax: Axes | None = None,
    zone_type: int = 1,
    n_phi: int = 0,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    linestyle: dict[str, Any] = LINE_STYLE,
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
    linestyle
        line style, by default ``{"color": "black", "linewidth": 0.25}``

    Returns
    -------
        tuple of matplotlib figure and axes objects

    Examples
    --------
    .. prompt:: python >>> auto

        >>> plot_grids_rz(zone_type=1, n_phi=10)

    .. image:: ../_static/images/plotting/plot_grids_rz.png
    """
    rmin, rmax, zmin, zmax = rz_range
    if rmin >= rmax or zmin >= zmax:
        raise ValueError("Invalid rz_range")

    # set default line style
    if not isinstance(linestyle, dict):
        raise TypeError("linestyle must be a dict")

    linestyle.setdefault("color", "black")
    linestyle.setdefault("linewidth", 0.5)

    if not isinstance(ax, Axes):
        if not isinstance(fig, Figure):
            fig, ax = plt.subplots(dpi=200)
        else:
            ax = fig.add_subplot()

    ax.set_aspect("equal")

    if zone_type not in {1, 2}:
        raise ValueError(f"zone_type must be either 1 or 2. (zone_type: {zone_type})")

    zone_type -= 1

    for zone in ZONES[zone_type]:
        emc = Grid(zone=zone)
        L, M, _ = emc.shape

        # plot radial line
        if zone in {"zone0", "zone11"}:
            num_pol = M - 1
        else:
            num_pol = M
        for m in range(num_pol):
            ax.plot(
                emc.grid_data[:, m, n_phi, 0],
                emc.grid_data[:, m, n_phi, 1],
                **linestyle
            )
        # plot poloidal line
        for l in range(L):
            ax.plot(
                emc.grid_data[l, :, n_phi, 0],
                emc.grid_data[l, :, n_phi, 1],
                **linestyle
            )

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
    ax.set_xlabel("R[m]")
    ax.set_ylabel("Z[m]")

    return (fig, ax)


def install_tetra_meshes(
    zones: list[str] = ZONES[0] + ZONES[1],
    tetra_mesh_path: Path | str = DEFAULT_TETRA_MESH_PATH,
    update=True,
    **kwargs,
) -> None:
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
            emc = Grid(zone, **kwargs)

            # prepare vertices and tetrahedral indices
            vertices = emc.generate_vertices()
            tetrahedra = tetrahedralize(emc.generate_cell_indices())

            # create TetraMesh instance (heavy calculation)
            tetra = TetraMesh(vertices, tetrahedra, tolerant=False)

            # save
            tetra.save(tetra_path)

            sp.ok()


if __name__ == "__main__":
    # debug
    grid = Grid("zone0")
    pass
