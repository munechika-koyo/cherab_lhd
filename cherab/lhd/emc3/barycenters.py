"""Module to handle barycenters derived from EMC3-EIRENE grids."""

from pathlib import Path
from types import EllipsisType

import h5py  # noqa: F401
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..tools.fetch import fetch_file
from .repository.utility import path_validate

__all__ = ["CenterGrid"]


class CenterGrid:
    """Class for center grids of EMC3-EIRENE grids.

    One EMC3-based cell is divided to six tetrahedra and the center point of each cell is defined
    as the average of the six tetrahedra's barycenters.
    Considering various indexing ways, final center points are averaged by integrating several
    cells, which must have 3 dimensional resolutions w.r.t. radial/poloidal/toroidal direction.

    | Total number of center grid is L x M x N, each letter of which means:
    | L: Radial grid resolution
    | M: Poloidal grid resolution
    | N: Toroidal grid resolution.

    Parameters
    ----------
    zone : {"zone0", "zone11"}
        Name of zone, currently supporting a few zones.
    index_type : {"coarse", "cell"}
        Indexing way of center grids, by default ``"coarse"``.
    dataset : str, optional
        Name of dataset, by default ``"emc3/grid-360.nc"``.
    grid_file : Path | str | None, optional
        Path to the grid file. If specified, the grid dataset is loaded from the file preferentially.
        Otherwise, the grid file is fetched from the repository.
    **kwargs
        Keyword arguments to pass to `.fetch_file`.

    Examples
    --------
    >>> cgrid = CenterGrid("zone0", index_type="cell")
    >>> cgrid
    CenterGrid(zone='zone0', index_type='cell', dataset='/path/to/cache/cherab/lhd/emc3/grid-360.nc')
    >>> str(cgrid)
    'CenterGrid with cell index_type (zone: zone0, L: 82, M: 601, N: 37)'
    """

    def __init__(
        self,
        zone: str,
        index_type: str = "coarse",
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

        # Load center points dataArray into memory
        with xr.open_dataset(path, group=f"{zone}/centers/{index_type}") as ds:
            self._da = ds["center_points"].load()

        # Set properties
        self._zone = zone
        self._index_type = index_type
        self._path = path
        self._shape = self._da.shape[0], self._da.shape[1], self._da.shape[2]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(zone={self.zone!r}, index_type={self.index_type!r}, "
            f"dataset={self._path!r})"
        )

    def __str__(self) -> str:
        L, M, N = self._shape
        return (
            f"{self.__class__.__name__} for (zone: {self.zone}, index_type: {self.index_type}, "
            f"L: {L}, M: {M}, N: {N})"
        )

    def __getitem__(
        self, key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | NDArray
    ) -> NDArray[np.float64] | float:
        """Return center grid coordinates indexed by (l, m, n, xyz).

        Returned grid coordinates are in :math:`(X, Y, Z)` which can be specified by
        ``l``: radial, ``m``: poloidal, ``n``: torodial indices.

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="cell")
        >>> cgrid[0, 0, 0, :]  # (l=0, m=0, n=0)
        array([ 3.59664909e+00,  7.84665944e-03, -5.75750000e-04])  # (x, y, z)

        >>> cgrid[:, -10, 0, :]  # (radial coords at m=-10, n=0)
        array([[3.59672601e+00, 7.84684125e-03, 1.13558333e-03],
                [3.57695347e+00, 7.80372411e-03, 1.03814167e-02],
                ...
                [3.26883531e+00, 7.13347363e-03, 1.63643583e-01]])
        """
        return self._da.data[key]

    @property
    def zone(self) -> str:
        """Name of zone.

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="cell")
        >>> cgrid.zone
        'zone0'
        """
        return self._zone

    @property
    def path(self) -> str:
        """Path to dataset.

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="cell")
        >>> cgrid.path
        '/path/to/cache/cherab/lhd/emc3/grid-360.nc'
        """
        return self._path

    @property
    def index_type(self) -> str:
        """Indexing way of center grids.

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="cell")
        >>> cgrid.index_type
        'cell'
        """
        return self._index_type

    @property
    def data_array(self) -> xr.DataArray:
        """`~.xarray.DataArray` of center grid coordinates.

        The data array has 4 dimensions, which are (ρ, θ, ζ, XYZ).
        ρ, θ, ζ are radial, poloidal, toroidal coordinates, respectively.
        They are normalized to edge of the grid, so that the range is (0, 1).
        XYZ are X, Y, Z coordinates [m].

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="coarse")
        >>> cgrid.data_array
        <xarray.DataArray 'center_points' (ρ: 33, θ: 100, ζ: 9, ΧΥΖ: 3)> Size: 713kB
        array([[[[ 3.59639165e+00,  3.13846353e-02, -2.37611806e-03],
                 [ 3.59458235e+00,  9.41257225e-02, -5.97601736e-03],
                 [ 3.59100504e+00,  1.56783858e-01, -9.40089757e-03],
                 ...,
                 [ 3.16185746e+00,  3.60201716e-01, -2.63552739e-02],
                 [ 3.12687989e+00,  4.11609016e-01, -2.80624431e-02],
                 [ 3.08802431e+00,  4.61450838e-01, -2.50325625e-02]]]])
        Coordinates:
        * ρ        (ρ) float64 264B 0.01515 0.04545 0.07576 ... 0.9242 0.9545 0.9848
        * θ        (θ) float64 800B 0.005 0.015 0.025 0.035 ... 0.975 0.985 0.995
        * ζ        (ζ) float64 72B 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5
        * ΧΥΖ      (ΧΥΖ) <U1 12B 'X' 'Y' 'Z'
        Attributes:
            units:      m
            long_name:  center points
        """
        return self._da

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of center grids.

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="cell")
        >>> cgrid.shape
        (82, 601, 36)
        """
        return self._shape

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """Array of center grid coordinates of each volume.

        The dimension of array is 4 dimension, shaping ``(L, M, N, 3)``.
        The coordinate is :math:`(X, Y, Z)` [m].

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="cell")
        >>> grid.grid_data.shape
        (81, 600, 36, 3)
        >>> grid.grid_data
        array([[[[ 3.59664909e+00,  7.84665938e-03, -5.75750000e-04],
                [ 3.59653587e+00,  2.35395361e-02, -1.49250000e-03],
                [ 3.59631043e+00,  3.92310971e-02, -2.40650000e-03],
                ...,
                [ 3.07201514e+00,  4.52253492e-01, -6.34334583e-02],
                [ 3.06137608e+00,  4.64343114e-01, -6.15580417e-02],
                [ 3.05057222e+00,  4.76330154e-01, -5.93228750e-02]]]])
        """
        return self._da.data

    def get_lmn(self, index) -> tuple[int, int, int]:
        """Return (l, m, n) indices from 1D index.

        (l, m, n) means radial, poloidal and toroidal indices, respectively.

        Parameters
        ----------
        index : int
            1D index corresponding to center grid.

        Returns
        -------
        tuple[int, int, int]
            (l, m, n) indices.

        Examples
        --------
        >>> cgrid = CenterGrid("zone0", index_type="cell")
        >>> cgrid.get_lmn(0)
        (0, 0, 0)
        >>> cgrid.get_lmn(100)
        (19, 1, 0)
        """
        L, M, _ = self._shape
        return index % L, (index // L) % M, index // (L * M)
