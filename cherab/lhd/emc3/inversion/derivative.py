"""Construct derivative matrices of the EMC3-EIRENE grids."""

from functools import cached_property
from itertools import product
from typing import Literal

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, lil_matrix

from ..barycenters import CenterGrid
from ..curvilinear import CurvCoords
from .polygon import generate_boundary_map

__all__ = [
    "Derivative",
    "create_dmats_pairs_subdomains",
]


class Derivative:
    """Class for derivative matrices.

    This class is used to represent derivative matrices of the EMC3-EIRENE-defined center grids,
    and calculate radial, poloidal and toroidal derivative matrices.

    This derivative matrices follows the radial (:math:`\\rho`), poloidal (:math:`\\theta`) and
    toroidal (:math:`\\zeta`) directions of the magnetic field lines, which are based on
    the EMC3-EIRENE-defined center grids.

    Parameters
    ----------
    grid : `.CenterGrid`
        `.CenterGrid` instance of the EMC3-EIRENE-defined center grids.
    diff_type : {"forward", "central"}, optional
        Numerical differentiation type for radial and poloidal direction.
        The default is "forward", which means the forward difference method is used to calculate
        the derivative matrices. The other option is and "central".
    """

    def __init__(
        self, grid: CenterGrid, diff_type: Literal["forward", "central"] = "forward"
    ) -> None:
        self.grid = grid
        self.diff_type = diff_type

    @property
    def grid(self) -> CenterGrid:
        """EMC3-EIRENE-defined center grids."""
        return self._grid

    @grid.setter
    def grid(self, grid: CenterGrid) -> None:
        if not isinstance(grid, CenterGrid):
            raise TypeError(f"{grid=} must be an instance of {CenterGrid=}")
        self._grid = grid

    @property
    def diff_type(self) -> str:
        """Numerical differentiation type."""
        return self._diff_type

    @diff_type.setter
    def diff_type(self, diff_type: str) -> None:
        if diff_type not in {"forward", "central"}:
            raise ValueError(f"{diff_type=} must be one of 'forward' or 'central'")
        self._diff_type = diff_type

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
    def dmat_rho(self) -> csr_matrix:
        """Radial (:math:`\\rho`: direction) derivative matrix.

        The poloidal derivative matrix is constructed by the numerical difference method.
        """
        L, M, N = self.grid.shape

        dmat = lil_matrix((L * M * N, L * M * N), dtype=np.float64)

        # memoryview
        index = self.index.view()

        match self._diff_type:
            case "central":
                for m, n in np.ndindex(M, N):
                    # calculate length of each segment along to rho direction
                    length = np.linalg.norm(
                        self.grid[1:, m, n, :] - self.grid[0:-1, m, n, :], axis=1
                    )

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

            case "forward":
                for m, n in np.ndindex(M, N):
                    # calculate length of each segment along to rho direction
                    length = np.linalg.norm(
                        self.grid[1:, m, n, :] - self.grid[0:-1, m, n, :], axis=1
                    )

                    # border condition at l = L - 1 with dirichlet condition
                    # TODO: implement connection between other zones
                    dmat[index[-1, m, n], index[-1, m, n]] = -1 / length[-1]

                    for l in range(0, L - 1):
                        dmat[index[l, m, n], index[l, m, n]] = -1 / length[l]
                        dmat[index[l, m, n], index[l + 1, m, n]] = 1 / length[l]

        return dmat.tocsr()

    @cached_property
    def dmat_theta(self) -> csr_matrix:
        """Poloidal (:math:`\\theta`: direction) derivative matrix.

        The poloidal derivative matrix is constructed by the numerical difference method.
        """
        L, M, N = self.grid.shape

        dmat = lil_matrix((L * M * N, L * M * N), dtype=np.float64)

        # memoryview
        index = self.index.view()

        match self._diff_type:
            case "central":
                for l, n in np.ndindex(L, N):
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

            case "forward":
                for l, n in np.ndindex(L, N):
                    if self.grid.zone in {"zone0", "zone11"}:
                        # connect the last point to the first point
                        grid = np.vstack((self.grid[l, :, n, :], self.grid[l, 0, n, :]))

                        # calculate length of each segment along to theta direction
                        length = np.linalg.norm(grid[1:, :] - grid[0:-1, :], axis=1)

                        # border condition at m = M - 1
                        dmat[index[l, -1, n], index[l, -1, n]] = -1 / length[-1]
                        dmat[index[l, -1, n], index[l, 0, n]] = 1 / length[-1]

                    else:
                        # calculate length of each segment along to theta direction
                        length = np.linalg.norm(
                            self.grid[l, 1:, n, :] - self.grid[l, :-1, n, :], axis=1
                        )

                        # border condition at m = M - 1 with dirichlet condition
                        dmat[index[l, -1, n], index[l, -1, n]] = -1 / length[-1]

                    for m in range(0, M - 1):
                        dmat[index[l, m, n], index[l, m, n]] = -1 / length[m]
                        dmat[index[l, m, n], index[l, m + 1, n]] = 1 / length[m]

        return dmat.tocsr()

    @cached_property
    def dmat_zeta(self) -> csr_matrix:
        """Toroidal (:math:`\\zeta`: direction) derivative matrix.

        This derivative matrix is calculated by using the forward difference method.
        """
        L, M, N = self.grid.shape

        dmat = lil_matrix((L * M * N, L * M * N), dtype=np.float64)

        # memoryview
        index = self.index.view()

        for l, m in np.ndindex(L, M):
            # calculate length of each segment along to theta direction
            length = np.linalg.norm(self.grid[l, m, 1:, :] - self.grid[l, m, 0:-1, :], axis=1)

            # connection between subdomains
            if self.grid.zone in {"zone0", "zone1", "zone2", "zone3", "zone4"}:
                # border condition at n = N - 1
                # This depends on the connection to the next subdomain, so
                # it is done by other function.
                pass

            elif self.grid.zone in {"zone11", "zone12", "zone13", "zone14", "zone15"}:
                # border condition at n = N - 1
                # backward difference
                dmat[index[l, m, -1], index[l, m, -2]] = -1 / length[-1]
                dmat[index[l, m, -1], index[l, m, -1]] = 1 / length[-1]

            else:
                raise NotImplementedError("Connection to back subdomains is not implemented.")

            for n in range(0, N - 1):
                # forward difference
                dmat[index[l, m, n], index[l, m, n]] = -1 / length[n]
                dmat[index[l, m, n], index[l, m, n + 1]] = 1 / length[n]

        return dmat.tocsr()

    def create_dmats_pairs(
        self, mode: Literal["strict", "ii", "flux"] = "strict"
    ) -> list[tuple[csr_matrix, csr_matrix]]:
        """Create derivative matrices for each coordinate pair.

        Parameters
        ----------
        mode : {"strict", "ii", "flux"}, optional
            Derivative matrix mode, by default "strict"

        Returns
        -------
        list[tuple[:obj:`~scipy.sparse.csr_matrix`, :obj:`~scipy.sparse.csr_matrix`]]
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


def create_dmats_pairs_subdomains(
    zone1: str = "zone0",
    zone2: str = "zone11",
    index_type: str = "coarse",
    mode: Literal["strict", "ii", "ii+no-metric", "flux"] = "strict",
    min_points: int = 200,
    ratio: float = 1.0,
) -> list[tuple[csr_matrix, csr_matrix]]:
    """Create derivative matrices for each coordinate pair considering the connection between two
    subdomains.

    This function is used to conbaine two derivative matrices both of which are connected along
    the toroidal direction like zone0 and zone11.

    Parameters
    ----------
    zone1 : str, optional
        Zone name of the first subdomain, by default "zone0".
    zone2 : str, optional
        Zone name of the second subdomain, by default "zone11".
    index_type : str, optional
        Index type, by default "coarse".
    mode : {"strict", "ii", "ii+no-metric", "flux"}, optional
        Derivative matrix mode, by default "strict".
    min_points : int, optional
        Minimum number of points to be used for the boundary map, by default 200.
    ratio : float, optional
        Ratio of the number of points to be used for the boundary map, by default 1.0.

    Returns
    -------
    list[tuple[:obj:`~scipy.sparse.csr_matrix`, :obj:`~scipy.sparse.csr_matrix`]]
        List of derivative matrices for each coordinate pair.
    """
    # generate boundary map
    bmap = generate_boundary_map(
        zone1, zone2, index_type=index_type, min_points=min_points, ratio=ratio
    ).tocsr()

    grid1 = CenterGrid(zone1, index_type=index_type)
    grid2 = CenterGrid(zone2, index_type=index_type)

    deriv1 = Derivative(grid1)
    deriv2 = Derivative(grid2)

    curv1 = CurvCoords(grid1)
    curv2 = CurvCoords(grid2)

    dmats1 = [deriv1.dmat_rho, deriv1.dmat_theta, deriv1.dmat_zeta]
    dmats2 = [deriv2.dmat_rho, deriv2.dmat_theta, deriv2.dmat_zeta]

    bases1 = [curv1.b_sup_rho, curv1.b_sup_theta, curv1.b_sup_zeta]
    bases2 = [curv2.b_sup_rho, curv2.b_sup_theta, curv2.b_sup_zeta]

    # combine two derivative matrices
    dmats = []
    for dmat1, dmat2 in zip(dmats1, dmats2, strict=True):
        dmat = bmat([[dmat1, None], [None, dmat2]], format="csr")
        dmats.append(dmat)

    # pop the toroidal derivative matrix
    dmat_zeta = dmats.pop(2).tolil()
    n_rows = dmats1[0].shape[0]

    # first index at n = -1
    first_index = deriv1.index[0, 0, -1]
    last_index = deriv1.index[-1, -1, -1]

    for i in range(first_index, last_index + 1):
        row_data = bmap[i - first_index, :]
        weights = row_data.data
        cols = row_data.indices

        total_coeff = 0

        for j, weight in zip(cols, weights, strict=True):
            l1, m1, n1 = grid1.get_lmn(i)
            l2, m2, n2 = grid2.get_lmn(j)

            # calculate length of each segment along to theta direction
            length = np.linalg.norm(grid1[l1, m1, n1, :] - grid2[l2, m2, n2, :])

            # insert coefficient (weight / length) to zone2 area
            dmat_zeta[i, j + n_rows] = weight / length

            # sum up coefficients
            total_coeff += weight / length

        # insert coefficient (-total_coeff) to zone1 area
        dmat_zeta[i, i] = -total_coeff

    # reverse dmat_zeta to dmats
    dmats.append(dmat_zeta.tocsr())

    results = []

    match mode:
        case "strict":
            product_list = list(product(range(3), repeat=2))

            for i, j in product_list:
                metric1 = diags(curv1.compute_metric(bases1[i], bases1[j]).ravel(order="F"))
                metric2 = diags(curv2.compute_metric(bases2[i], bases2[j]).ravel(order="F"))
                metric = bmat([[metric1, None], [None, metric2]])
                results.append((dmats[i], metric @ dmats[j]))  # (D_i, G^ij * D_j)

        case "flux":
            raise NotImplementedError("Flux coord drivative matrix is not implemented yet.")

        case "ii":
            for i in range(3):
                metric1 = diags(curv1.compute_metric(bases1[i], bases1[i]).ravel(order="F"))
                metric2 = diags(curv2.compute_metric(bases2[i], bases2[i]).ravel(order="F"))
                metric = bmat([[metric1, None], [None, metric2]])
                results.append((dmats[i], metric @ dmats[i]))  # (D_i, G^ii * D_i)

        case "ii+no-metric":
            for i in range(3):
                results.append((dmats[i], dmats[i]))  # (D_i, D_i)

        case _:
            raise ValueError(f"Invalid mode: {mode}")

    return results
