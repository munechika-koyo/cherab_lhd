"""
Module to deal with EMC3-EIRENE-defined grids
"""
import json
import os

import numpy as np
from cherab.lhd.tools.visualization import set_axis_properties
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

__all__ = ["EMC3Grid"]


BASE = os.path.dirname(__file__)
GRID_BASE_PATH = os.path.join(BASE, "data", "grid-360")
CELLGEO_PATH = os.path.join(BASE, "data", "CELL_GEO.pickle")

ZONES = [
    ["zone0", "zone1", "zone2", "zone3", "zone4"],  # zone_type = 1
    ["zone11", "zone12", "zone13", "zone14", "zone15"],  # zone_type 2
]
# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6


class EMC3Grid:
    """
    Grid vertices and cell indices generation of EMC3-EIRENE.
    This class offers methods to produce EMC3 grid vertices in :math:`(X, Y, Z)` coordinates
    and cell indices representing a cubic-like mesh with 8 vertices.
    Using these data, procedure of generating a :obj:`~raysect.primitive.mesh.tetra_mesh.TetraMesh`
    instance is also implemented.

    Total number of grids coordinates is L x M x N:<br>
    L: Radial grid resolution<br>
    M: Poloidal grid resolution<br>
    N: Toroidal grid resolution


    Parameters
    ----------
    zone
        name of grid zone. Users can select only one option of ``"zone0" - "zone21"``.
        Note that corresponding ``zone#.npy`` grid data must be stored in the directory ``path``.
    grid_directory
        directory path where grid data (e.g. ``zone0.npy``) are stored, by default
        ``../emc3/data/grid-360/``

    Examples
    --------
    .. prompt:: python >>> auto

        >>> grid = EMC3Grid("zone0")
        >>> grid

    """

    def __init__(self, zone: str, grid_directory: str = GRID_BASE_PATH) -> None:
        # === Parameters validation ================================================================
        # set and validate grid stored directory
        if not isinstance(grid_directory, str):
            raise TypeError("grid_directory must be string")
        if not os.path.exists(grid_directory):
            raise FileNotFoundError(f"{grid_directory} does not exists")
        self._grid_directory = grid_directory

        # set and validate zone name
        if not isinstance(zone, str):
            raise TypeError("zone must be string")
        grid_data_path = os.path.join(grid_directory, f"grid-{zone}.npy")
        if not os.path.exists(grid_data_path):
            raise FileNotFoundError(f"grid-{zone}.npy file does not exits in {self.grid_directory}")
        self._zone = zone

        # === Load Grid Configuration ==============================================================
        # check if grid_config.json file exists
        grid_config_path = os.path.join(grid_directory, "grid_config.json")
        if not os.path.exists(grid_config_path):
            raise FileNotFoundError(f"{grid_config_path} does not exists.")

        # load grid config
        with open(grid_config_path, "r") as file:
            grid_config = json.load(file)
        self._grid_config = grid_config[zone]

        # === Load Grid data =======================================================================
        self._grid_data = np.load(grid_data_path)

    def __repr__(self) -> str:
        L = self.grid_config["L"]
        M = self.grid_config["M"]
        N = self.grid_config["N"]
        msg = f"EMC3-EIRENE Grid instance (L: {L}, M: {M}, N: {N})\n"
        return msg

    @property
    def grid_directory(self) -> str:
        """directory name to store grid data"""
        return self._grid_directory

    @property
    def zone(self) -> str:
        """name of zone"""
        return self._zone

    @property
    def grid_config(self) -> dict[str, int]:
        """configuration dictionary containing grid resolutions and number of cells

        .. prompt:: python >>> auto

            >>> grid = EMC3Grid("zone0")
            >>> grid.grid_config
            {'L': 81, 'M': 601, 'N': 37, 'num_cells': 1728000}
        """
        return self._grid_config

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """
        Raw Grid array data.
        This array is directly loaded from ``.npy`` file.
        The dimension of array is 3D, shaping ``(L * M, 3, N)``.
        The coordinate is :math:`(R, Z, \\phi)`.

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

    def generate_vertices(self) -> NDArray[np.float64]:
        """Generate grid vertices array.
        A `grid_data` array is converted to 2D array which represents a vertex in :math:`(X, Y, Z)`
        coordinates.

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
        n_phi: int = 0,
        rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    ):
        """Plotting EMC3-EIRENE-defined grids in :math:`r - z` plane.

        Parameters
        ----------
        fig, optional
            matplotlib figure object, by default ``plt.subplots(dpi=200)``
        n_phi, optional
            index of toroidal grid, by default 0
        rz_range
            sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
            by default ``(2.0, 5.5, -1.6, 1.6)``

        Returns
        -------
            tuple of matplotlib figure and axes object
        """
        rmin, rmax, zmin, zmax = rz_range
        if rmin >= rmax or zmin >= zmax:
            raise ValueError("Invalid rz_range")

        if not isinstance(fig, Figure):
            fig, ax = plt.subplots(dpi=200)
        else:
            ax = fig.add_axes()

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
                linewidth=0.08,
                color="#7d7d7d",
            )
        # plot poloidal line
        for l in range(L):
            ax.plot(
                self.grid_data[l : L * M : L, 0, n_phi],
                self.grid_data[l : L * M : L, 1, n_phi],
                linewidth=0.08,
                color="#7d7d7d",
            )

        ax.set_xlim(rmin, rmax)
        ax.set_ylim(zmin, zmax)

        ax.text(
            rmin + (rmax - rmin) * 0.02,
            zmax - (zmax - zmin) * 0.02,
            f"$\\phi=${self.grid_data[0, 2, n_phi]:.1f}$^\\circ$",
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
    zone_type: int = 1,
    n_phi: int = 0,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
) -> tuple[Figure, Axes]:
    """Plotting EMC-EIRENE-defined grids in :math:`r - z` plane.

    Parameters
    ----------
    fig, optional
        matplotlib figure object, by default ``plt.subplots(dpi=200)``
    zone_type, optional
        type of zones collections, by default 1
        type 1 is ``["zone0", "zone1", "zone2", "zone3", "zone4"]``,
        type 2 is ``["zone11", "zone12", "zone13", "zone14", "zone15"]``.
    n_phi, optional
        index of toroidal grid, by default 0
    rz_range
        sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``

    Returns
    -------
        tuple of matplotlib figure and axes objects
    """
    rmin, rmax, zmin, zmax = rz_range
    if rmin >= rmax or zmin >= zmax:
        raise ValueError("Invalid rz_range")

    if not isinstance(fig, Figure):
        fig, ax = plt.subplots(dpi=200)
    else:
        ax = fig.add_axes()

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
                linewidth=0.08,
                color="#7d7d7d",
            )
        # plot poloidal line
        for l in range(L):
            ax.plot(
                emc.grid_data[l : L * M : L, 0, n_phi],
                emc.grid_data[l : L * M : L, 1, n_phi],
                linewidth=0.08,
                color="#7d7d7d",
            )

    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)

    ax.text(
        rmin + (rmax - rmin) * 0.02,
        zmax - (zmax - zmin) * 0.02,
        f"$\\phi=${emc.grid_data[0, 2, n_phi]:.1f}$^\\circ$",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
    )
    set_axis_properties(ax)
    ax.set_xlabel("R[m]")
    ax.set_ylabel("Z[m]")

    return (fig, ax)


if __name__ == "__main__":
    # debug
    grid = EMC3Grid("zone0")
    pass
