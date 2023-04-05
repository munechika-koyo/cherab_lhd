"""Module to deal with EMC3-EIRENE-defined grids."""
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from raysect.primitive.mesh import TetraMesh

from ..tools.spinner import Spinner
from ..tools.visualization import set_axis_properties
from .cython import tetrahedralize
from .repository.utility import DEFAULT_HDF5_PATH, DEFAULT_TETRA_MESH_PATH

__all__ = ["EMC3Grid", "plot_grids_rz", "install_tetra_meshes"]

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
LINE_STYLE = {"color": "#7d7d7d", "linewidth": 0.5}


class EMC3Grid:
    """Grid vertices and cell indices generation of EMC3-EIRENE.

    This class offers methods to produce EMC3 grid vertices in :math:`(X, Y, Z)` coordinates and
    cell indices representing a cubic-like mesh with 8 vertices.
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

        >>> grid = EMC3Grid("zone0")
        >>> grid
        EMC3-EIRENE Grid instance (zone: zone0, L: 82, M: 601, N: 37)
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
        self._zone = zone

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
        L = self.grid_config["L"]
        M = self.grid_config["M"]
        N = self.grid_config["N"]
        msg = f"EMC3-EIRENE Grid instance (zone: {self.zone}, L: {L}, M: {M}, N: {N})\n"
        return msg

    @property
    def hdf5_path(self) -> Path:
        """HDF5 dataset file path."""
        return self._hdf5_path

    @property
    def zone(self) -> str:
        """name of zone."""
        return self._zone

    @property
    def grid_config(self) -> dict[str, int]:
        """configuration dictionary containing grid resolutions and number of
        cells.

        .. prompt:: python >>> auto

            >>> grid = EMC3Grid("zone0")
            >>> grid.grid_config
            {'L': 81, 'M': 601, 'N': 37, 'num_cells': 1728000}
        """
        return self._grid_config

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """Raw Grid coordinates data array.

        The dimension of array is 3D, shaping ``(L * M, 3, N)``.
        The coordinate is :math:`(R, Z, \\phi)`. :math:`\\phi` is in [degree].

        .. prompt:: python >>> auto

            >>> grid = EMC3Grid("zone0")
            >>> grid.grid_data.shape
            (48681, 3, 37)
            >>> grid.grid_data
            array([[[ 3.593351e+00,  3.593307e+00,  3.593176e+00, ...,
                      3.551275e+00,  3.549266e+00,  3.547254e+00],
                    [-0.000000e+00, -1.835000e-03, -3.667000e-03, ...,
                     -4.099100e-02, -4.103600e-02, -4.099800e-02],
                    [ 0.000000e+00,  2.500000e-01,  5.000000e-01, ...,
                      8.500000e+00,  8.750000e+00,  9.000000e+00]],
                    ...
                   [[ 3.262600e+00,  3.262447e+00,  3.261987e+00, ...,
                      3.096423e+00,  3.087519e+00,  3.078508e+00],
                    [ 0.000000e+00, -4.002000e-03, -7.995000e-03, ...,
                     -7.012100e-02, -6.796400e-02, -6.543900e-02],
                    [ 0.000000e+00,  2.500000e-01,  5.000000e-01, ...,
                      8.500000e+00,  8.750000e+00,  9.000000e+00]]])
        """
        return self._grid_data

    def index(self, l: int, m: int, n: int) -> NDArray[np.float64]:
        """Return grid coordinates indexed by (l, m, n).

        Parameters
        ----------
        l
            index of radial direction
        m
            index of poloidal direction
        n
            index of toroidal direction

        Returns
        -------
        :obj:`~numpy.ndarray`
            grid coordinates (3, ) array in :math:`(R, Z, \\phi)`.
        """
        L = self.grid_config["L"]
        M = self.grid_config["M"]
        N = self.grid_config["N"]

        if not (0 <= l < L and 0 <= m < M and 0 <= n < N):
            raise ValueError(
                f"Invalid grid index (l, m, n) = ({l}, {m}, {n}). "
                f"Each index must be in [0, {L}), [0, {M}), [0, {N})."
            )

        return self.grid_data[L * m + l, :, n]

    def generate_vertices(self) -> NDArray[np.float64]:
        """Generate grid vertices array. A `grid_data` array is converted to 2D
        array which represents a vertex in :math:`(X, Y, Z)` coordinates.

        Returns
        -------
        :obj:`~numpy.ndarray`
            vertices (L x N x M, 3) 2D array


        .. prompt:: python >>> auto

            >>> grid = EMC3Grid("zone0")
            >>> verts = grid.generate_vertices()
            >>> verts.shape
            (1801197, 3)
            >>> verts
            array([[ 3.593351  ,  0.        , -0.        ],
                   [ 3.559212  ,  0.        , -0.        ],
                   [ 3.526526  ,  0.        , -0.        ],
                   ...,
                   [ 3.04105882,  0.4816564 , -0.065465  ],
                   [ 3.04083165,  0.48162042, -0.065452  ],
                   [ 3.04060646,  0.48158475, -0.065439  ]])
        """
        L, M, N = self.grid_config["L"], self.grid_config["M"], self.grid_config["N"]
        vertices = np.zeros((L * M * N, 3), dtype=np.float64)
        grid = self._grid_data.view()
        for n in range(N):
            phi = np.deg2rad(grid[0, 2, n])
            for m in range(M):
                for l in range(L):
                    row = M * L * n + L * m + l
                    vertices[row, 0] = grid[L * m + l, 0, n] * np.cos(phi)
                    vertices[row, 1] = grid[L * m + l, 0, n] * np.sin(phi)
                    vertices[row, 2] = grid[L * m + l, 1, n]

        return vertices

    def generate_cell_indices(self) -> NDArray[np.uint32]:
        """Generate cell indices array.

        One row of cell indices array represents one cubic-like mesh with 8 vertices.

        Returns
        -------
        :obj:`~numpy.ndarray`
            cell indices ((L-1) x (M-1) x (N-1), 8) 2D array


        .. prompt:: python >>> auto

            >>> grid = EMC3Grid("zone0")
            >>> cells = grid.generate_cell_indices()
            >>> cells.shape
            (1728000, 8)
            >>> cells
            array([[      0,       1,      82, ...,   48682,   48763,   48762],
                   [      1,       2,      83, ...,   48683,   48764,   48763],
                   [      2,       3,      84, ...,   48684,   48765,   48764],
                   ...,
                   [1752431, 1752432, 1752513, ..., 1801113, 1801194, 1801193],
                   [1752432, 1752433, 1752514, ..., 1801114, 1801195, 1801194],
                   [1752433, 1752434, 1752515, ..., 1801115, 1801196, 1801195]],
                  dtype=uint32)
        """
        L, M, N = self.grid_config["L"], self.grid_config["M"], self.grid_config["N"]
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
        line: dict[str, Any] = LINE_STYLE,
    ):
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
        line
            line style, by default ``{"color": "#7d7d7d", "linewidth": 0.5}``

        Returns
        -------
            tuple of matplotlib figure and axes object


        .. prompt:: python >>> auto

            >>> grid = EMC3Grid("zone0")
            >>> grid.plot()

        .. image:: ../_static/images/plotting/grid_zone0.png
        """
        rmin, rmax, zmin, zmax = rz_range
        if rmin >= rmax or zmin >= zmax:
            raise ValueError("Invalid rz_range")

        # set default line style
        if not isinstance(line, dict):
            raise TypeError("line must be a dict")

        line.setdefault("color", "#7d7d7d")
        line.setdefault("width", 0.5)

        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots(dpi=200)
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        L = self.grid_config["L"]
        M = self.grid_config["M"]

        # plot radial line
        if self.zone in {"zone0", "zone11"}:
            num_pol = M - 1
        else:
            num_pol = M
        for m in range(num_pol):
            start = m * L
            ax.plot(
                self.grid_data[start : start + L, 0, n_phi],
                self.grid_data[start : start + L, 1, n_phi],
                linewidth=line["width"],
                color=line["color"],
            )
        # plot poloidal line
        for l in range(L):
            ax.plot(
                self.grid_data[l : L * M : L, 0, n_phi],
                self.grid_data[l : L * M : L, 1, n_phi],
                linewidth=line["width"],
                color=line["color"],
            )

        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)

        ax.text(
            rmin + (rmax - rmin) * 0.02,
            zmax - (zmax - zmin) * 0.02,
            f"$\\phi=${self.grid_data[0, 2, n_phi]:.2f}$^\\circ$",
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
    line: dict[str, Any] = LINE_STYLE,
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
    line
        line style, by default ``{"color": "#7d7d7d", "linewidth": 0.5}``

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
    if not isinstance(line, dict):
        raise TypeError("line must be a dict")

    line.setdefault("color", "#7d7d7d")
    line.setdefault("width", 0.5)

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
        emc = EMC3Grid(zone=zone)
        L = emc.grid_config["L"]
        M = emc.grid_config["M"]

        # plot radial line
        if zone in {"zone0", "zone11"}:
            num_pol = M - 1
        else:
            num_pol = M
        for m in range(num_pol):
            start = m * L
            ax.plot(
                emc.grid_data[start : start + L, 0, n_phi],
                emc.grid_data[start : start + L, 1, n_phi],
                linewidth=line["width"],
                color=line["color"],
            )
        # plot poloidal line
        for l in range(L):
            ax.plot(
                emc.grid_data[l : L * M : L, 0, n_phi],
                emc.grid_data[l : L * M : L, 1, n_phi],
                linewidth=line["width"],
                color=line["color"],
            )

    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)

    ax.text(
        rmin + (rmax - rmin) * 0.02,
        zmax - (zmax - zmin) * 0.02,
        f"$\\phi=${emc.grid_data[0, 2, n_phi]:.2f}$^\\circ$",
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
    **kwargs : :obj:`.EMC3Grid` properties, optional
        *kwargs* are used to specify :obj:`.EMC3Grid` properties except for ``zone`` argument.
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
            emc = EMC3Grid(zone, **kwargs)

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
    grid = EMC3Grid("zone0")
    pass
