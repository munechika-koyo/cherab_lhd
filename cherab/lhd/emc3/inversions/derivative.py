"""Construct derivative matrices of the EMC3-EIRENE grids."""
from functools import cached_property

import numpy as np
from scipy.sparse import csr_array, lil_matrix

from ..barycenters import CenterGrids


class Derivative:
    """Class for derivative matrices.

    This class is used to represent derivative matrices of the EMC3-EIRENE-defined center grids,
    and calculate radial, poloidal and toroidal derivative matrices.

    This derivative matrices follows the radial, poloidal and toroidal directions of the
    magnetic field lines, which are based on the EMC3-EIRENE-defined center grids.

    Parameters
    ----------
    grid
        :obj:`.CenterGrids` object.
    """
    def __init__(self, grid: CenterGrids) -> None:
        self.grid = grid

        # create index array
        L, M, N = self.grid.shape
        self.index = np.arange(L * M * N, dtype=np.uint32).reshape(L, M, N, order="F")

    @property
    def grid(self) -> CenterGrids:
        """EMC3-EIRENE-defined center grids."""
        return self._grid

    @grid.setter
    def grid(self, grid: CenterGrids) -> None:
        if not isinstance(grid, CenterGrids):
            raise TypeError(f"{grid=} must be an instance of {CenterGrids=}")
        self._grid = grid

    @cached_property
    def radial_dmat(self) -> csr_array:
        """Radial derivative matrix."""
        L, M, N = self.grid.shape

        dmat = lil_matrix((L * M * N, L * M * N), dtype=np.float64)

        # memoryview
        index = self.index.view()

        for n in range(N):
            for m in range(M):
                # calculate length of each segment along to rho direction
                length = np.linalg.norm(self.grid[1:, m, n, :] - self.grid[0:-1, m, n, :], axis=1)

                # TODO: implement connection between other zones
                # border condition at l = 0 with forward difference
                dmat[index[0, m, n], index[1, m, n]] = 1 / length[0]
                dmat[index[0, m, n], index[0, m, n]] = -1 / length[0]

                # border condition at l = L - 1 with dirichlet condition
                dmat[index[-1, m, n], index[-2, m, n]] = -0.5

                for l in range(1, L - 1):
                    denom = length[l - 1] * length[l] * (length[l - 1] + length[l])

                    dmat[index[l, m, n], index[l - 1, m, n]] = -(length[l] ** 2) / denom
                    dmat[index[l, m, n], index[l + 0, m, n]] = (
                        length[l] ** 2 - length[l - 1] ** 2
                    ) / denom
                    dmat[index[l, m, n], index[l + 1, m, n]] = length[l - 1] ** 2 / denom

        return dmat.tocsr()

    @cached_property
    def poloidal_dmat(self) -> csr_array:
        """Poloidal derivative matrix."""
        L, M, N = self.grid.shape

        dmat = lil_matrix((L * M * N, L * M * N), dtype=np.float64)

        # memoryview
        index = self.index.view()

        for n in range(N):
            for l in range(L):
                # connect the last point to the first point
                grid = np.vstack((self.grid[l, :, n, :], self.grid[l, 0, n, :]))

                # calculate length of each segment along to theta direction
                length = np.linalg.norm(grid[1:, :] - grid[0:-1, :], axis=1)

                # border condition at m = 0 and m = M - 1
                # TODO: implement border condition except for zone0 & zone11
                for m in [0, M - 1]:
                    denom = length[m - 1] * length[m] * (length[m - 1] + length[m])

                    dmat[index[l, m, n], index[l, m - 1, n]] = -(length[m] ** 2) / denom
                    dmat[index[l, m, n], index[l, m + 0, n]] = (
                        length[m] ** 2 - length[m - 1] ** 2
                    ) / denom
                    dmat[index[l, m, n], index[l, 0, n]] = length[m - 1] ** 2 / denom

                for m in range(1, M - 1):
                    denom = length[m - 1] * length[m] * (length[m - 1] + length[m])

                    dmat[index[l, m, n], index[l, m - 1, n]] = -(length[m] ** 2) / denom
                    dmat[index[l, m, n], index[l, m + 0, n]] = (
                        length[m] ** 2 - length[m - 1] ** 2
                    ) / denom
                    dmat[index[l, m, n], index[l, m + 1, n]] = length[m - 1] ** 2 / denom

        return dmat.tocsr()

    @cached_property
    def toroidal_dmat(self) -> csr_array:
        """Toroidal derivative matrix."""
        L, M, N = self.grid.shape

        dmat = lil_matrix((L * M * N, L * M * N), dtype=np.float64)

        # memoryview
        index = self.index.view()

        for m in range(M):
            for l in range(L):
                # calculate length of each segment along to theta direction
                length = np.linalg.norm(self.grid[l, m, 1:, :] - self.grid[l, m, 0:-1, :], axis=1)

                # TODO: implement connection between subdomains
                # border condition at n = 0
                dmat[index[l, m, 0], index[l, m, 1]] = 1 / length[0]
                dmat[index[l, m, 0], index[l, m, 0]] = -1 / length[0]

                # border condition at n = N - 1
                dmat[index[l, m, -1], index[l, m, -2]] = -1 / length[-1]
                dmat[index[l, m, -1], index[l, m, -1]] = 1 / length[-1]

                for n in range(1, N - 1):
                    denom = length[n - 1] * length[n] * (length[n - 1] + length[n])

                    dmat[index[l, m, n], index[l, m, n - 1]] = -(length[n] ** 2) / denom
                    dmat[index[l, m, n], index[l, m, n + 0]] = (
                        length[n] ** 2 - length[n - 1] ** 2
                    ) / denom
                    dmat[index[l, m, n], index[l, m, n + 1]] = length[n - 1] ** 2 / denom

        return dmat.tocsr()
