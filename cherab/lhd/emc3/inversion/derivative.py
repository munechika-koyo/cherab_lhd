"""Construct derivative matrices of the EMC3-EIRENE grids."""
from functools import cached_property
from itertools import product

import numpy as np
from scipy.sparse import csr_array, diags, lil_array

from ...tools.spinner import Spinner
from ..barycenters import CenterGrids
from ..curvilinear import CurvCoords


class Derivative:
    """Class for derivative matrices.

    This class is used to represent derivative matrices of the EMC3-EIRENE-defined center grids,
    and calculate radial, poloidal and toroidal derivative matrices.

    This derivative matrices follows the radial (:math:`\\rho`), poloidal (:maht:`\\theta`) and
    toroidal (:math:`\\zeta`) directions of the magnetic field lines, which are based on
    the EMC3-EIRENE-defined center grids.

    Parameters
    ----------
    grid
        :obj:`.CenterGrids` object.
    """

    def __init__(self, grid: CenterGrids) -> None:
        self.grid = grid

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
    def index(self) -> np.ndarray:
        """Index of the EMC3-EIRENE-defined center grids.

        The shape of the index array is (L, M, N), which follows the order of (:math:`\\rho`,
        :math:`\\theta`, :math:`\\zeta`) grid resolution.

        The index array is used to convert the 3D grid coordinates indices to 1D array index, i.e.
        ``index[l, m, n] = i`` means the 3D grid coordinates ``(l, m, n)`` is converted to the 1D
        array index ``i``.
        """
        L, M, N = self.grid.shape
        return np.arange(L * M * N, dtype=np.uint32).reshape(L, M, N, order="F")

    @cached_property
    def dmat_rho(self) -> csr_array:
        """Radial (:math:\\rho: direction) derivative matrix."""
        L, M, N = self.grid.shape

        dmat = lil_array((L * M * N, L * M * N), dtype=np.float64)

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
    def dmat_theta(self) -> csr_array:
        """Poloidal (:math:\\theta: direction) derivative matrix."""
        L, M, N = self.grid.shape

        dmat = lil_array((L * M * N, L * M * N), dtype=np.float64)

        # memoryview
        index = self.index.view()

        for n in range(N):
            for l in range(L):
                # connect the last point to the first point
                grid = np.vstack((self.grid[l, :, n, :], self.grid[l, 0, n, :]))

                # calculate length of each segment along to theta direction
                length = np.linalg.norm(grid[1:, :] - grid[0:-1, :], axis=1)

                # TODO: implement border condition except for zone0 & zone11
                for m in range(M):
                    denom = length[m - 1] * length[m] * (length[m - 1] + length[m])

                    dmat[index[l, m, n], index[l, m - 1, n]] = -(length[m] ** 2) / denom
                    dmat[index[l, m, n], index[l, m + 0, n]] = (
                        length[m] ** 2 - length[m - 1] ** 2
                    ) / denom

                    # border condition at m = M - 1
                    if m == M - 1:
                        dmat[index[l, m, n], index[l, 0, n]] = length[m - 1] ** 2 / denom
                    else:
                        dmat[index[l, m, n], index[l, m + 1, n]] = length[m - 1] ** 2 / denom

        return dmat.tocsr()

    @cached_property
    def dmat_zeta(self) -> csr_array:
        """Toroidal (:math:\\zeta: direction) derivative matrix."""
        L, M, N = self.grid.shape

        dmat = lil_array((L * M * N, L * M * N), dtype=np.float64)

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

    @Spinner(text="Creating derivative matrices...", timer=True)
    def create_dmats_pairs(self, mode: str = "strict") -> list[tuple[csr_array, csr_array]]:
        """Create derivative matrices for each coordinate pair.

        Parameters
        ----------
        mode : str, optional
            Derivative matrix mode, by default "strict"

        Returns
        -------
        list[tuple[csr_array, csr_array]]
            List of derivative matrices for each coordinate pair.
        """
        curv = CurvCoords(self.grid)

        results = []

        match mode:
            case "strict":
                product_list = list(product(range(3), repeat=2))
                bases = [curv.b_sup_rho, curv.b_sup_theta, curv.b_sup_zeta]
                dmats = [self.dmat_rho, self.dmat_theta, self.dmat_zeta]

                for i, j in product_list:
                    metric = diags(curv.compute_metric(bases[i], bases[j]).ravel(order="F"))
                    results.append((dmats[i], metric @ dmats[j]))  # (D_i, G^ij * D_j)

            case "flux":
                raise NotImplementedError("Flux coord drivative matrix is not implemented yet.")

            case "ii":
                bases = [curv.b_sup_rho, curv.b_sup_theta, curv.b_sup_zeta]
                dmats = [self.dmat_rho, self.dmat_theta, self.dmat_zeta]

                for i in range(3):
                    metric = diags(curv.compute_metric(bases[i], bases[i]).ravel(order="F"))
                    results.append((dmats[i], metric @ dmats[i]))  # (D_i, G^ii * D_i)

            case _:
                raise ValueError(f"Invalid mode: {mode}")

        return results
