"""A module that handles barycenters derived from EMC3-EIRENE grids."""
cimport numpy as np
cimport cython
from cython.parallel import prange

from pathlib import Path
from types import EllipsisType

import h5py
import numpy as np
from numpy.typing import NDArray

from .cython.tetrahedralization cimport tetrahedralize
from .grid cimport Grid

from .repository.utility import DEFAULT_HDF5_PATH

__all__ = ["CenterGrids"]


cdef class CenterGrids:
    """Class for dealing with barycenter coordinates of variouse EMC3-EIRENE-based volume.

    One EMC3 cell is divided to six tetrahedra and the center point of each cell is defined
    as the avarage of the six tetrahedra's barycenters.
    Considering variouse indexing ways, final center points are averaged by integrating several
    cells, which must have 3 dimensional resolutions w.r.t. radial/poloidal/toroidal.

    | Total number of center grid is L x M x N, each letter of which means:
    | L: Radial grid resolution
    | M: Poloidal grid resolution
    | N: Toroidal grid resolution.

    Parameters
    ----------
    zone
        name of zone
    index_type
        indexing way of center grids selected from [`"cell"`] as far as implemented,
        by default ``"cell"``
    grid_group
        name of grid group, by default ``"grid-360"``
    hdf5_path
        path to the HDF5 file storing grid dataset, by default ``~/.cherab/lhd/emc3.hdf5``.


    Examples
    --------
    .. prompt:: python >>> auto

        >>> cgrid = CenterGrids("zone0", index_type="cell")
        >>> cgrid
        CenterGrids(zone='zone0', index_type='cell', grid_group='grid-360')
        >>> str(cgrid)
        'CenterGrids with cell index_type (zone: zone0, L: 82, M: 601, N: 37, number of cells: 1749600)'
    """

    def __init__(
        self,
        zone: str,
        index_type: str = "cell",
        grid_group: str = "grid-360",
        hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    ) -> None:
        cdef:
            object file, dset

        # set and check HDF5 file path
        if isinstance(hdf5_path, (Path, str)):
            self._hdf5_path = Path(hdf5_path)
        else:
            raise TypeError("hdf5_path must be a string or a pathlib.Path instance.")
        if not self._hdf5_path.exists():
            raise FileNotFoundError(f"{self._hdf5_path.name} file does not exist.")

        # set properties
        self._zone = zone
        self._grid_group = grid_group
        self._index_type = index_type

        # load center coordinates from HDF5 file
        try:
            with h5py.File(self._hdf5_path, mode="r") as file:
                # Load center grids dataset
                dset = file[f"{grid_group}/{zone}/centers/{index_type}"]

                # Load center grids coordinates
                self._grid_data = dset[:]

                # Load grid configuration
                self._config = GridConfig(
                    L=dset.attrs["L"],
                    M=dset.attrs["M"],
                    N=dset.attrs["N"],
                    total=dset.attrs["total"],
                )
        except KeyError:
            # Generate center grids if they have not been stored in the HDF5 file
            self._grid_data, self._config = self.generate(zone, index_type, grid_group=grid_group)

        # set shape
        self._shape = self._config.L, self._config.M, self._config.N

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(zone={self.zone!r}, index_type={self.index_type!r}, "
            f"grid_group={self.grid_group!r})"
        )

    def __str__(self) -> str:
        L, M, N, total = (
            self._config.L,
            self._config.M,
            self._config.N,
            self._config.total
        )
        return (
            f"{self.__class__.__name__} for (zone: {self.zone}, index_type: {self.index_type}, "
            f"L: {L}, M: {M}, N: {N}, number of cells: {total})"
        )

    def __getitem__(
        self, key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | NDArray
    ) -> NDArray[np.float64] | float:
        """Return center grid coordinates indexed by (l, m, n, xyz).

        Returned grid coordinates are in :math:`(X, Y, Z)` which can be specified by
        ``l``: radial, ``m``: poloidal, ``n``: torodial indices.

        Examples
        --------
        .. prompt:: python >>> auto

            >>> cgrid = CenterGrids("zone0")
            >>> cgrid[0, 0, 0, :]  # (l=0, m=0, n=0)
            array([ 3.59664909e+00,  7.84665944e-03, -5.75750000e-04])  # (x, y, z)

            >>> cgrid[:, -10, 0, :]  # (radial coords at m=-10, n=0)
            array([[3.59672601e+00, 7.84684125e-03, 1.13558333e-03],
                   [3.57695347e+00, 7.80372411e-03, 1.03814167e-02],
                   ...
                   [3.26883531e+00, 7.13347363e-03, 1.63643583e-01]])
        """
        return self._grid_data[key]

    @property
    def zone(self) -> str:
        """Name of zone."""
        return self._zone

    @property
    def index_type(self) -> str:
        """Indexing way of center grids."""
        return self._index_type

    @property
    def grid_group(self) -> str:
        """Name of grid group."""
        return self._grid_group

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of grid (L, M, N)."""
        return self._shape

    @property
    def config(self) -> dict[str, int]:
        """Configuration dictionary containing grid resolutions and total number.

        .. prompt:: python >>> auto

            >>> cgrid = CenterGrids("zone0")
            >>> cgrid.config
            {'L': 81, 'M': 600, 'N': 36, 'total': 1749600}
        """
        return {
            "L": self._config.L,
            "M": self._config.M,
            "N": self._config.N,
            "total": self._config.total
        }

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """Array of center grid coordinates of each volume.

        The dimension of array is 4 dimension, shaping ``(L, M, N, 3)``.
        The coordinate is :math:`(X, Y, Z)` [m].

        .. prompt:: python >>> auto

            >>> cgrid = CenterGrid("zone0")
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
        return self._grid_data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef (int, int, int) get_lmn(self, int index):
        """Return (l, m, n) indices from 1D index.

        (l, m, n) means radial, poloidal and toroidal indices, respectively.

        Parameters
        ----------
        index
            1D index

        Returns
        -------
        tuple[int, int, int]
            (l, m, n) indices
        """
        cdef int L, M

        L, M, _ = self._shape
        return index % L, (index // L) % M, index // (L * M)

    @classmethod
    def generate(
        cls, zone: str, index_type: str, grid_group: str = "grid-360"
    ) -> tuple[NDArray[np.float64], GridConfig]:
        """Generate the center grids for a given zone.

        If corresponding center grids have not been stored in the HDF5 file,
        they will be generated and stored.

        Parameters
        ----------
        zone
            The zone to generate the center grids for.
        index_type
            The indexing way of center grids selected from [`"cell"`] as far as implemented.
        grid_group
            name of grid group corresponding to magnetic axis configuration, by default ``grid-360``.

        Returns
        -------
        tuple[NDArray[np.float64], int, int, int]
            The center grids for the given zone. Each tuple element is: (grids array, L, M, N).
        """
        cdef:
            object file, index_group, zone_group, centers_group, dset
            np.ndarray indices, grid
            GridConfig config

        # retrieve cell indexing array from HDF5 file
        with h5py.File(DEFAULT_HDF5_PATH, mode="r") as file:
            # get index group
            index_group = file[f"{grid_group}/{zone}/index"]

            if index_type not in index_group:
                msg = f"{index_type} indexing is not implemented."
                raise KeyError(msg)
            else:
                indices = index_group[index_type][:]

        # compute center grids
        grid, config = _compute_centers(zone, grid_group, indices)

        # store center grids in HDF5 file
        with h5py.File(DEFAULT_HDF5_PATH, mode="r+") as file:
            # get zone group
            zone_group = file[f"{grid_group}/{zone}"]

            # create centers group if it does not exist
            if "centers" not in zone_group:
                centers_group = zone_group.create_group("centers")
            else:
                centers_group = zone_group["centers"]

            # delete existing center grids
            if index_type in centers_group:
                del centers_group[index_type]

            dset = centers_group.create_dataset(name=index_type, data=grid)
            dset.attrs["L"] = config.L
            dset.attrs["M"] = config.M
            dset.attrs["N"] = config.N
            dset.attrs["total"] = config.total

        return (grid, config)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef object _compute_centers(str zone, str grid_group, np.ndarray indices):
    """Compute center grids from vertices and indices.

    Parameters
    ----------
    zone
        name of zone
    grid_group
        name of grid group
    indices
        indexing array
    Returns
    -------
    numpy.ndarray (L, M, N, 3)
        center grids
    """
    cdef:
        Grid grid
        np.ndarray vertices, cells, tetrahedra, verts, verts_rec, count_array
        np.float64_t[::1, :] vertices_mv
        np.float64_t[:, ::1] verts_mv
        np.uint32_t[:, ::1] tetrahedra_mv
        np.uint32_t[:, :, ::1] indices_mv
        np.float64_t[:, ::1] verts_rec_mv
        np.uint32_t[::1] count_array_mv
        int num_cell, i, j, k, L, M, N, L_new, M_new, N_new, total, index

    # Create cells and tetrahedra from Grid
    grid = Grid(zone, grid_group=grid_group)
    vertices = grid.generate_vertices()
    cells = grid.generate_cell_indices()
    tetrahedra = tetrahedralize(cells)

    vertices_mv = vertices
    tetrahedra_mv = tetrahedra
    indices_mv = indices

    num_cell = cells.shape[0]

    # calculate center of each cell (with original resolution)
    verts = np.zeros((num_cell, 3), dtype=float)
    verts_mv = verts

    # with divide by 6 x 4 for one cell
    for i in prange(num_cell, nogil=True):
        for j in range(6 * i, 6 * (i + 1)):
            for k in tetrahedra_mv[j, :]:
                verts_mv[i, 0] += vertices_mv[k, 0] / 24.0
                verts_mv[i, 1] += vertices_mv[k, 1] / 24.0
                verts_mv[i, 2] += vertices_mv[k, 2] / 24.0

    # convert 1-D to 4-D array
    L = indices_mv.shape[0]
    M = indices_mv.shape[1]
    N = indices_mv.shape[2]

    # reconstruct centers considering specific indexing way
    L_new = indices_mv[L - 1, 0, 0] + 1
    M_new = (indices_mv[L - 1, M - 1, 0] + 1) // L_new
    N_new = (indices_mv[L - 1, M - 1, N - 1] + 1) // (L_new * M_new)
    total = L_new * M_new * N_new

    # calculate center of each cell (with specific resolution)
    verts_rec = np.zeros((total, 3), dtype=float)
    verts_rec_mv = verts_rec

    # count up for each cell if it is included in the specific resolution
    count_array = np.zeros(total, dtype=np.uint32)
    count_array_mv = count_array

    for i in range(num_cell):
        # get l, m, n indices from 1D index
        l, m, n = i % L, (i // L) % M, i // (L * M)

        # get an index from a specific indexing array
        index = indices_mv[l, m, n]

        # add each center to the reconstructed center
        verts_rec_mv[index, 0] += verts_mv[i, 0]
        verts_rec_mv[index, 1] += verts_mv[i, 1]
        verts_rec_mv[index, 2] += verts_mv[i, 2]

        # count up for each cell if it is included in the specific resolution
        count_array_mv[index] += 1

    # divide by the number of cells included in the specific resolution
    for i in range(total):
        verts_rec_mv[i, 0] /= <double>count_array_mv[i]
        verts_rec_mv[i, 1] /= <double>count_array_mv[i]
        verts_rec_mv[i, 2] /= <double>count_array_mv[i]

    return (
        verts_rec.reshape((L_new, M_new, N_new, 3), order="F"),
        GridConfig(L=L_new, M=M_new, N=N_new, total=total),
    )
