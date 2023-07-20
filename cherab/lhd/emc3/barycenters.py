"""A module that handles barycenters derived from EMC3-EIRENE grids."""
from pathlib import Path
from types import EllipsisType

import h5py
import numpy as np
from numpy.typing import NDArray

from ..tools.spinner import Spinner
from .cython import tetrahedralize
from .grid import Grid
from .repository.utility import DEFAULT_HDF5_PATH

__all__ = ["CenterGrids"]


class CenterGrids:
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
    index
        indexing way of center grids selected from [`"cell"`] as far as implemented,
        by default ``"cell"``
    grid_group
        name of grid group, by default ``"grid-360"``
    hdf5_path
        path to the HDF5 file storing grid dataset, by default ``~/.cherab/lhd/emc3.hdf5``.


    Examples
    --------
    .. prompt:: python >>> auto

        >>> grid = CenterGrids("zone0", index="cell")
        >>> grid
        CenterGrids(zone='zone0', index='cell', grid_group='grid-360')
        >>> str(grid)
        'CenterGrids with cell index (zone: zone0, L: 82, M: 601, N: 37, number of cells: 1749600)'
    """

    def __init__(
        self,
        zone: str,
        index="cell",
        grid_group: str = "grid-360",
        hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    ) -> None:
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
        self._index = index

        # load center coordinates from HDF5 file
        try:
            with h5py.File(self._hdf5_path, mode="r") as h5file:
                # Load center grids dataset
                dset = h5file[f"{grid_group}/{zone}/centers/{index}"]

                # Load center grids coordinates
                self._grid_data = dset[:]

                # Load grid configuration
                self._grid_config = dict(
                    L=dset.attrs["L"],
                    M=dset.attrs["M"],
                    N=dset.attrs["N"],
                    total=dset.attrs["total"],
                )
        except KeyError:
            # Generate center grids if they have not been stored in the HDF5 file
            self._grid_data, L, M, N = self.generate(zone, index, grid_group=grid_group)
            # Set grid configuration
            self._grid_config = dict(
                L=L,
                M=M,
                N=N,
                total=L * M * N,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zone={self.zone!r}, grid_group={self.grid_group!r})"

    def __str__(self) -> str:
        L, M, N, total = self._grid_config.values()
        return f"{self.__class__.__name__} for (zone: {self.zone}, L: {L}, M: {M}, N: {N}, total: {total})"

    def __getitem__(
        self, key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | NDArray
    ) -> NDArray[np.float64] | float:
        """Return center grid coordinates indexed by (l, m, n, xyz).

        Examples
        --------
        .. prompt:: python >>> auto

            >>> grid = CenterGrids("zone0")
            >>> grid[0, 0, 0, :]  # (l=0, m=0, n=0)
            array([ 3.59664909e+00,  7.84665944e-03, -5.75750000e-04])  # (x, y, z)

            >>> grid[:, -10, 0, :]  # (radial coords at m=-10, n=0)
            array([[3.59672601e+00, 7.84684125e-03, 1.13558333e-03],
                   [3.57695347e+00, 7.80372411e-03, 1.03814167e-02],
                   ...
                   [3.26883531e+00, 7.13347363e-03, 1.63643583e-01]])
        """
        return self.grid_data[key]

    @property
    def zone(self) -> str:
        """Name of zone."""
        return self._zone

    @property
    def index(self) -> str:
        """Indexing way of center grids."""
        return self._index

    @property
    def grid_group(self) -> str:
        """Name of grid group."""
        return self._grid_group

    @property
    def grid_config(self) -> dict:
        """Grid configuration."""
        return self._grid_config

    @property
    def grid_data(self) -> NDArray[np.float64]:
        """Array of center grid coordinates of each volume."""
        return self._grid_data

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of grid (L, M, N)."""
        return self.grid_config["L"], self.grid_config["M"], self.grid_config["N"]

    @classmethod
    def generate(
        cls, zone: str, index: str, grid_group: str = "grid-360"
    ) -> tuple[NDArray[np.float64], int, int, int]:
        """Generate the center grids for a given zone.

        If corresponding center grids have not been stored in the HDF5 file,
        they will be generated and stored.

        Parameters
        ----------
        zone
            The zone to generate the center grids for.
        index
            The indexing way of center grids selected from [`"cell"`] as far as implemented.
        grid_group
            name of grid group corresponding to magnetic axis configuration, by default ``grid-360``.

        Returns
        -------
        tuple[NDArray[np.float64], int, int, int]
            The center grids for the given zone. Each tuple element is: (grids array, L, M, N).
        """
        with Spinner(f"Generating center points of {zone}'s cells...", timer=True) as sp:
            # retrieve cell indexing array from HDF5 file
            with h5py.File(DEFAULT_HDF5_PATH, mode="r") as h5file:
                # get index group
                index_group = h5file[f"{grid_group}/{zone}/index"]

                if index not in index_group:
                    msg = f"{index} indexing is not implemented."
                    sp.fail()
                    raise KeyError(msg)
                else:
                    indices = index_group[index][:]
                    L, M, N = indices.shape

            # generate vertices, cells and tetrahedra
            grid = Grid(zone, grid_group=grid_group)
            vertices = grid.generate_vertices()
            cells = grid.generate_cell_indices()
            tetrahedra = tetrahedralize(cells)

            # calculate center of each cell
            verts = np.zeros((cells.shape[0], 3), dtype=float)
            for i in range(cells.shape[0]):
                for j in range(6 * i, 6 * (i + 1)):
                    for k in tetrahedra[j, :]:
                        verts[i] += vertices[k, :]

            # divide by 6 x 4 for one cell
            verts /= 24

            # convert 1-D to 4-D array
            verts = verts.reshape((L, M, N, 3), order="F")

            # reconstruct centers considering indexing way
            # TODO: implement other indexing ways

            # store center grids in HDF5 file
            with h5py.File(grid.hdf5_path, mode="r+") as h5file:
                # get zone group
                zone_group = h5file[f"{grid_group}/{zone}"]

                # create centers group if it does not exist
                if "centers" not in zone_group:
                    centers_group = zone_group.create_group("centers")
                else:
                    centers_group = zone_group["centers"]

                # delete existing center grids
                if index in centers_group:
                    del centers_group[index]

                dset = centers_group.create_dataset(name=index, data=verts)
                dset.attrs["L"] = L
                dset.attrs["M"] = M
                dset.attrs["N"] = N
                dset.attrs["total"] = L * M * N

            sp.ok()

        return (verts, L, M, N)
