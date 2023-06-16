"""Module for dealing with barycenters of EMC3 cells."""
import h5py
import numpy as np
from numpy.typing import NDArray

from ..tools.spinner import Spinner
from .cython import tetrahedralize
from .grid import EMC3Grid
from .repository.utility import DEFAULT_HDF5_PATH


class EMC3CenterGrids:
    """Class for dealing with barycenters of EMC3-EIRENE cells.

    One EMC3 cell is divided to six tetrahedra and the center point of each cell is defined
    as the avarage of the six tetrahedra's barycenters. Each center point is ordered by
    the radial, poloidal and toroidal-like coordinates.

    | Total number of grids is L x M x N, each letter of which means:
    | L: Radial grid resolution
    | M: Poloidal grid resolution
    | N: Toroidal grid resolution.

    Parameters
    ----------
    zone
        name of zone
    grid_group
        name of grid group, by default ``"grid-360"``
    """

    def __init__(self, zone: str, grid_group: str = "grid-360") -> None:
        self._zone = zone
        self._grid_group = grid_group

        # load center coordinates from HDF5 file
        try:
            with h5py.File(DEFAULT_HDF5_PATH, mode="r") as h5file:
                # Load center grids dataset
                dset = h5file[f"{grid_group}/{zone}/centers"]

                # Load center grids coordinates
                self._centers = dset[:]

                # Load grid configuration
                self._grid_config = dict(
                    L=dset.attrs["L"],
                    M=dset.attrs["M"],
                    N=dset.attrs["N"],
                    total=dset.attrs["total"],
                )
        except KeyError:
            self._centers, L, M, N = self.generate(zone, grid_group=grid_group)
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

    @property
    def zone(self) -> str:
        """Name of zone."""
        return self._zone

    @property
    def grid_group(self) -> str:
        """Name of grid group."""
        return self._grid_group

    @property
    def grid_config(self) -> dict:
        """Grid configuration."""
        return self._grid_config

    @property
    def centers(self) -> NDArray[np.float64]:
        """Array of center coordinates of each cell."""
        return self._centers

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of grid (L, M, N)."""
        return self.grid_config["L"], self.grid_config["M"], self.grid_config["N"]

    def index(self, l: int, m: int, n: int) -> NDArray[np.float64]:
        """Return the center coordinates of a cell for a given index.

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
        NDArray[np.float64]
            The center coordinates of a cell for the given index.
            coordinates is :math:`(X, Y, Z)`.
        """
        L, M, N = self.shape

        if not (0 <= l < L and 0 <= m < M and 0 <= n < N):
            raise ValueError(
                f"Invalid grid index (l, m, n) = ({l}, {m}, {n}). "
                f"Each index must be in [0, {L}), [0, {M}), [0, {N})."
            )

        return self.centers[l + m * L + n * L * M, :]

    @classmethod
    def generate(
        cls, zone: str, grid_group: str = "grid-360"
    ) -> tuple[NDArray[np.float64], int, int, int]:
        """Generate the center grids for a given zone.

        If the center grids have not been stored in the HDF5 file, they will be generated and stored.

        Parameters
        ----------
        zone
            The zone to generate the center grids for.
        grid_group
            name of grid group corresponding to magnetic axis configuration, by default ``grid-360``.

        Returns
        -------
        tuple[NDArray[np.float64], int, int, int]
            The center grids for the given zone.
        """
        with Spinner(f"Generating center points of {zone}'s cells...", timer=True) as sp:
            emc = EMC3Grid(zone, grid_group=grid_group)
            vertices = emc.generate_vertices()
            cells = emc.generate_cell_indices()
            tetrahedra = tetrahedralize(cells)

            # calculate center of each cell
            verts = np.zeros((cells.shape[0], 3), dtype=float)
            for i in range(cells.shape[0]):
                for j in range(6 * i, 6 * (i + 1)):
                    for k in tetrahedra[j, :]:
                        verts[i] += vertices[k, :]

                # divide by 6 x 4 for one cell
                verts[i] /= 24

            # store center grids in HDF5 file
            with h5py.File(emc.hdf5_path, mode="a") as h5file:
                # get existing grid group
                zone_group = h5file[grid_group][zone]

                # delete existing center grids dataset
                if "centers" in zone_group:
                    del zone_group["centers"]

                dset = zone_group.create_dataset(name="centers", data=verts)
                L = emc.grid_config["L"] - 1
                M = emc.grid_config["M"] - 1
                N = emc.grid_config["N"] - 1
                dset.attrs["L"] = L
                dset.attrs["M"] = M
                dset.attrs["N"] = N
                dset.attrs["total"] = cells.shape[0]

            sp.ok()

        return (verts, L, M, N)
