"""Module to deal with EMC3-EIRENE-defined grids."""

from pathlib import Path
from types import EllipsisType

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from numpy.typing import ArrayLike, NDArray

from ..machine.wall import periodic_toroidal_angle
from ..tools.fetch import PATH_TO_STORAGE, fetch_file
from ..tools.spinner import Spinner
from ..tools.visualization import add_inner_title
from .cython import TetraMeshData
from .cython.tetrahedralization import tetrahedralize
from .repository.utility import path_validate

__all__ = ["Grid", "install_tetra_meshes"]

ZONES: list = [
    ["zone0", "zone1", "zone2", "zone3", "zone4"],  # zone_type = 1
    ["zone11", "zone12", "zone13", "zone14", "zone15"],  # zone_type 2
]
# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6

# Plotting config.
LINE_STYLE: dict = {"color": "black", "linewidth": 0.5}


class Grid:
    """Class for dealing with grid coordinates defined by EMC3-EIRENE.

    This class handles originally defined EMC3-EIRENE grid coordinates in :math:`(R, Z)`,
    and offers methods to produce cell vertices in :math:`(X, Y, Z)` coordinates and
    their indices, which a **cell** means a cubic-like mesh with 8 vertices.
    Using these data, procedure of generating a `.TetraMeshData`
    instance is also implemented.

    | Total number of grids coordinates is L x M x N, each letter of which means:
    | L: Radial grid resolution
    | M: Poloidal grid resolution
    | N: Toroidal grid resolution.

    Parameters
    ----------
    zone : {"zone0", ..., "zone21"}
        Name of grid zone. Users can select only one option of ``"zone0"`` - ``"zone21"``.
    dataset : str, optional
        Name of dataset, by default ``"emc3/grid-360.nc"``.
    grid_file : Path | str | None, optional
        Path to the grid file. If specified, the grid dataset is loaded from the file preferentially.
        Otherwise, the grid file is fetched from the repository.
    **kwargs
        Keyword arguments to pass to `.fetch_file`.

    Examples
    --------
    >>> grid = Grid("zone0")
    >>> grid
    Grid(zone='zone0', dataset='/path/to/cache/cherab/lhd/emc3/grid-360.nc')
    >>> str(grid)
    'Grid for (zone: zone0, L: 82, M: 601, N: 37, number of cells: 1749600)'
    """

    def __init__(
        self,
        zone: str,
        dataset: str = "emc3/grid-360.nc",
        grid_file: str | Path | None = None,
        **kwargs,
    ) -> None:
        if grid_file is not None:
            # Manual input of grid file
            path = path_validate(grid_file)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
            if not path.suffix == ".nc":
                raise ValueError("grid_file must be a path to a NetCDF file.")
        else:
            # Fetch grid dataset
            path = fetch_file(dataset, **kwargs)

        # Load grid dataset with specified zone into memory
        with xr.open_dataset(path, group=zone) as ds:
            self._num_cells = ds.attrs["num_cells"]
            self._da = ds["grid"].load()

        # Set attributes
        self._zone = zone
        self._path = path
        self._shape = self._da.shape[0], self._da.shape[1], self._da.shape[2]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zone={self._zone!r}, dataset={self._path!r})"

    def __str__(self) -> str:
        L, M, N, num_cells = (
            *self._shape,
            self._num_cells,
        )
        return (
            f"{self.__class__.__name__} for (zone: {self.zone}, "
            f"L: {L}, M: {M}, N: {N}, number of cells: {num_cells})"
        )

    def __getitem__(
        self, key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | NDArray
    ) -> xr.DataArray:
        """Return grid coordinates indexed by (l, m, n, RZ).

        Returned grid coordinates are in :math:`(R, Z)` which can be specified by ``l``: radial,
        ``m``: poloidal, ``n``: torodial index.

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid[0, 0, 0, :].data  # (l=0, m=0, n=0)
        array([3.593351e+00, 0.000000e+00])  # (R, Z) coordinates

        >>> grid[:, -10, 0, :].data  # (radial coords at m=-10, n=0)
        array([[3.600000e+00, 0.000000e+00],
                [3.593494e+00, 3.076000e-03],
                [3.560321e+00, 1.875900e-02],
                ...,
                [3.267114e+00, 1.573770e-01]])
        """
        return self._da[key]

    @property
    def zone(self) -> str:
        """Name of zone.

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.zone
        'zone0'
        """
        return self._zone

    @property
    def path(self) -> str | Path:
        """Path to dataset.

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.path
        '/path/to/cache/cherab/lhd/emc3/grid-360.nc'
        """
        return self._path

    @property
    def data_array(self) -> xr.DataArray:
        """`~xarray.DataArray` instance.

        The data array has 4 dimensions, which are (ρ, θ, ζ, RZ).
        ρ, θ, ζ are radial, poloidal, toroidal coordinates, respectively.
        RZ is a coordinate for (R, Z).

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.data_array
        <xarray.DataArray 'grid' (ρ: 82, θ: 601, ζ: 37, RZ: 2)> Size: 29MB
        array([[[[ 3.600000e+00,  0.000000e+00],
                 [ 3.600000e+00,  0.000000e+00],
                 [ 3.600000e+00,  0.000000e+00],
                 ...
                 [ 3.096423e+00, -7.012100e-02],
                 [ 3.087519e+00, -6.796400e-02],
                 [ 3.078508e+00, -6.543900e-02]]]])
        Coordinates:
            * ρ      (ρ) float64 656B 0.0 0.01235 0.02469 0.03704 ... 0.9753 0.9877 1.0
            * θ      (θ) float64 5kB 0.0 0.001667 0.003333 0.005 ... 0.9967 0.9983 1.0
            * ζ      (ζ) float64 296B 0.0 0.25 0.5 0.75 1.0 ... 8.0 8.25 8.5 8.75 9.0
            * RZ     (RZ) <U1 8B 'R' 'Z'
        Attributes:
            units:      m
            long_name:  grid coordinates
        """
        return self._da

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of grid (L, M, N).

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.shape
        (82, 601, 37)
        """
        return self._shape

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """Raw Grid coordinates data array.

        The dimension of array is 4 dimension, shaping ``(L, M, N, 2)``.
        The coordinate is :math:`(R, Z)`.

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.grid_data.shape
        (82, 601, 37, 2)
        >>> grid.grid_data
        array([[[[ 3.600000e+00,  0.000000e+00],
                 [ 3.600000e+00,  0.000000e+00],
                 [ 3.600000e+00,  0.000000e+00],
                 ...
                 [ 3.096423e+00, -7.012100e-02],
                 [ 3.087519e+00, -6.796400e-02],
                 [ 3.078508e+00, -6.543900e-02]]]])
        """
        return self._da.data

    def generate_vertices(self) -> NDArray[np.float64]:
        """Generate grid vertices array. A `grid_data` array is converted to 2D array which
        represents a vertex in :math:`(X, Y, Z)` coordinates.

        The vertices array is stacked in the order of `(L, M, N)`.

        Returns
        -------
        `~numpy.ndarray`
            Vertices (L x N x M, 3) 2D array.

        Examples
        --------
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
        L, M, N = self._shape
        vertices = np.zeros((L, M, N, 3), dtype=np.float64)
        radians = self._da["ζ"].values * np.pi / 180.0
        grid_data = self._da.data

        for n, phi in enumerate(radians):
            vertices[..., n, 0] = grid_data[..., n, 0] * np.cos(phi)
            vertices[..., n, 1] = grid_data[..., n, 0] * np.sin(phi)
            vertices[..., n, 2] = grid_data[..., n, 1]

        return np.ascontiguousarray(vertices.reshape((L * M * N, 3), order="F"))

    def generate_cell_indices(self) -> NDArray[np.uint32]:
        """Generate cell indices array.

        One row of cell indices array represents one cubic-like mesh with 8 vertices.
        Cells are indexed in the order of `(L, M, N)` direction.

        Returns
        -------
        `~numpy.ndarray`
            Cell indices ((L-1) x (M-1) x (N-1), 8) 2D array.

        Examples
        --------
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
        L, M, N = self._shape
        cells = np.zeros(((L - 1) * (M - 1) * (N - 1), 8), dtype=np.uint32)
        cells_mv = cells

        i = 0
        for n, m, l in np.ndindex(N - 1, M - 1, L - 1):
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
        rz_range: tuple[float, float, float, float] | None = None,
        show_phi: bool = True,
        indices_radial: ArrayLike | slice | None = None,
        indices_poloidal: ArrayLike | slice | None = None,
        **kwargs,
    ) -> tuple[Figure | None, Axes]:
        """Plotting EMC3-EIRENE-defined grids in :math:`R–Z` plane.

        This method allows users to plot grid lines at a specific discretized toroidal angle.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`, optional
            Matplotlib figure object, by default ``fig = plt.figure()``.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes object, by default ``ax = fig.add_subplot()``.
        n_phi : int, optional
            Index of toroidal angle, by default 0.
        rz_range : tuple[float, float, float, float], optional
            Sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
            by default None.
        show_phi : bool
            Show toroidal angle text in the plot, by default True.
        indices_radial : array_like, slice, optional
            Specify indices of radial direction, by default None.
            If None, all radial grids are plotted.
        indices_poloidal : array_like, slice, optional
            Specify indices of poloidal direction, by default None.
            If None, all poloidal grids are plotted.
        **kwargs : dict
            `matplotlib.lines.Line2D` properties, by default
            ``{"color": "black", "linewidth": 0.5}``.

        Returns
        -------
        fig : `~matplotlib.figure.Figure` | None
            Matplotlib figure object. If ``fig`` is not specified but ``axis`` is, return None.
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes object.

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.plot(linewidth=0.2)

        .. image:: ../../_static/images/plotting/grid_zone0.png
        """
        if rz_range is not None:
            rmin, rmax, zmin, zmax = rz_range
            if rmin >= rmax or zmin >= zmax:
                raise ValueError("Invalid rz_range")

        # Extract grid data with indices
        if indices_radial is None:
            indices_radial = slice(0, None)
        if self._zone in {"zone0", "zone11"}:
            if indices_poloidal is None:
                indices_poloidal = slice(0, -1)
            elif hasattr(indices_poloidal, "__getitem__"):
                indices_poloidal = indices_poloidal[:-1]
            else:
                raise TypeError("indices_poloidal must be an 1-D array-like object.")
        else:
            if indices_poloidal is None:
                indices_poloidal = slice(0, None)

        grid_ρ = self._da.isel(θ=indices_poloidal)
        grid_θ = self._da.isel(ρ=indices_radial)

        # Set default line style
        for key, value in LINE_STYLE.items():
            kwargs.setdefault(key, value)

        # === plotting =============================================================================
        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        # Plot radial line
        ax.plot(
            grid_ρ[..., n_phi, 0],
            grid_ρ[..., n_phi, 1],
            **kwargs,
        )
        # Plot poloidal line
        ax.plot(
            grid_θ[..., n_phi, 0].T,
            grid_θ[..., n_phi, 1].T,
            **kwargs,
        )

        if rz_range is not None:
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(zmin, zmax)

        if show_phi:
            add_inner_title(ax, f"$\\phi=${self._da['ζ'][n_phi]:.2f}°")
        set_axis_properties(ax)

        return (fig, ax)

    def plot_coarse(self, **kwargs):
        """Plotting EMC-EIRENE-defined coarse grids in :math:`R–Z` plane.

        The indices to use as the coarse grid is stored in attributes of `"/index/coarse"` variables.
        So this method is available only if they are stored.

        Coarse grid indices are created by `.install_indices` function.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for `plot` method.

        Returns
        -------
        fig : `~matplotlib.figure.Figure` | None
            Matplotlib figure object. If ``fig`` is not specified but ``axis`` is, return None.
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes object.

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.plot_coarse()

        .. image:: ../../_static/images/plotting/grid_coarse_zone0.png
        """
        ds = xr.open_dataset(self._path, group=f"{self.zone}/index")["coarse"]
        indices_radial = ds.attrs["indices_radial"]
        indices_poloidal = ds.attrs["indices_poloidal"]
        return self.plot(indices_radial=indices_radial, indices_poloidal=indices_poloidal, **kwargs)

    def plot_outline(
        self,
        phi: float = 0.0,
        fig: Figure | None = None,
        ax: Axes | None = None,
        show_phi: bool = True,
        **kwargs,
    ) -> tuple[Figure | None, Axes]:
        """Plotting EMC3-EIRENE-defined grid outline in :math:`R–Z` plane.

        This method allows users to plot grid outline at a specific toroidal angle :math:`\\varphi`.
        The toroidal angle is arbitrary, where the grid outline is calculated by linear interpolation
        between two nearest toroidal grids.

        Parameters
        ----------
        phi : float, optional
            Toroidal grid in [degree], by default 0.0.
        fig : `~matplotlib.figure.Figure`, optional
            Matplotlib figure object, by default ``fig = plt.figure()``.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes object, by default ``ax = fig.add_subplot()``.
        show_phi : bool
            Show toroidal angle text in the plot, by default True.
        **kwargs : dict
            `matplotlib.lines.Line2D` properties, by default
            ``{"color": "black", "linewidth": 0.5}``.

        Returns
        -------
        fig : `~matplotlib.figure.Figure` | None
            Matplotlib figure object. If ``fig`` is not specified but ``axis`` is, return None.
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes object.

        Examples
        --------
        >>> grid = Grid("zone0")
        >>> grid.plot_outline(4.2)

        .. image:: ../../_static/images/plotting/grid_outline_zone0.png
        """
        # put phi in [0, 18) range
        phi_t, flipped = periodic_toroidal_angle(phi)
        phi_range = self._da["ζ"].isel(ζ=[0, -1]).data

        if phi_t < phi_range[0] or phi_t > phi_range[1]:
            raise ValueError(f"toroidal angle {phi_t} is out of grid range {phi_range}.")

        # set default line style
        for key, value in LINE_STYLE.items():
            kwargs.setdefault(key, value)

        if not isinstance(ax, Axes):
            if not isinstance(fig, Figure):
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot()

        ax.set_aspect("equal")

        if self.zone not in {"zone0", "zone11"}:
            # Interpolate grid
            outlines = self._da.interp(ζ=phi_t, method="linear")

            # flipped coords for z axis
            if flipped:
                outlines.loc[dict(RZ="Z")] *= -1

            # Plot poloidal line
            ax.plot(
                outlines.isel(ρ=[0, -1]).sel(RZ="R").T,
                outlines.isel(ρ=[0, -1]).sel(RZ="Z").T,
                **kwargs,
            )

            # Plot radial lines
            ax.plot(
                outlines.isel(θ=[0, -1]).sel(RZ="R"),
                outlines.isel(θ=[0, -1]).sel(RZ="Z"),
                **kwargs,
            )

        else:
            # Interpolate grid
            outlines = self._da.isel(ρ=-1).interp(ζ=phi_t, method="linear")

            # flipped coords for z axis
            if flipped:
                outlines.loc[dict(RZ="Z")] *= -1

            # plot outline (last poloidal line)
            ax.plot(outlines.sel(RZ="R"), outlines.sel(RZ="Z"), **kwargs)

        if show_phi:
            add_inner_title(ax, f"$\\phi=${phi:.2f}°")

        set_axis_properties(ax)

        return (fig, ax)


def set_axis_properties(axes: Axes):
    """Set x-, y-axis property. This function set axis labels and tickers.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        Matplotlib Axes object.
    """
    axes.set_xlabel("$R$ [m]")
    axes.set_ylabel("$Z$ [m]")
    axes.xaxis.set_minor_locator(MultipleLocator(0.1))
    axes.yaxis.set_minor_locator(MultipleLocator(0.1))
    axes.xaxis.set_major_formatter("{x:.1f}")
    axes.yaxis.set_major_formatter("{x:.1f}")
    axes.tick_params(direction="in", labelsize=10, which="both", top=True, right=True)


def install_tetra_meshes(
    zones: list[str] = ZONES[0] + ZONES[1],
    tetra_mesh_path: Path | str = PATH_TO_STORAGE / "emc3/tetra/",
    dataset: str = "grid-360",
    update: bool = True,
):
    """Create `.TetraMeshData` .rsm files and install them into a repository.

    Default repository is set to `.../cherab/lhd/emc3/tetra.`, which is located in the user's
    cache directory.
    The directory is created if it does not exist.
    The file name is determined by each zone name like ``zone0.rsm``.

    .. note::

        It takes a lot of time to calculate all `.TetraMeshData` instance because each zone has
        numerous number of grids.

    Parameters
    ----------
    zones : list[str], optional
        List of zone names, by default ``["zone0",..., "zone4", "zone11",..., "zone15"]``.
    tetra_mesh_path : Path | str, optional
        Path to the directory to save `.TetraMeshData` .rsm files,
        by default ``PATH_TO_STORAGE / "emc3/tetra/"``.
    dataset : str, optional
        Name of grid dataset, by default "grid-360".
    update : bool, optional
        Whether or not to update existing `.TetraMeshData` .rsm file, by default True.
    """
    tetra_mesh_path = path_validate(tetra_mesh_path)
    tetra_mesh_path.mkdir(parents=True, exist_ok=True)

    # populate each zone TetraMeshData instance
    for zone in zones:
        # path to the tetra .rsm file
        tetra_path = tetra_mesh_path / f"{zone}.rsm"

        with Spinner(text=f"Constructing {zone} tetra mesh...", timer=True) as sp:
            # skip if it exists and update is False
            if tetra_path.exists() and not update:
                sp.ok("⏩")
                continue

            # Load EMC3 grid
            emc = Grid(zone, dataset=dataset)

            # prepare vertices and tetrahedral indices
            vertices = emc.generate_vertices()
            tetrahedra = tetrahedralize(emc.generate_cell_indices())

            # create TetraMeshData instance (heavy calculation)
            tetra = TetraMeshData(vertices, tetrahedra, tolerant=False)

            # save
            tetra.save(tetra_path)

            sp.ok()
