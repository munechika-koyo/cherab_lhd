"""Module for handling curvilinear coordinates of EMC3-EIRENE-defined center grids."""

from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from .barycenters import CenterGrid

__all__ = ["CurvCoords"]


class CurvCoords:
    """Class for curvilinear coordinates.

    This class is used to represent curvilinear coordinates of EMC3-EIRENE-defined center grids, and
    calculate co/contra-variant bases, metric tensors, etc.

    This curvilinear coordinates follows the radial, poloidal and toroidal directions of the
    magnetic field lines, which are based on the EMC3-EIRENE-defined center grids.

    Parameters
    ----------
    grid : `.CenterGrid`
        Instance of `.CenterGrid` class.
    """

    def __init__(self, grid: CenterGrid):
        if not isinstance(grid, CenterGrid):
            raise TypeError(f"grid must be a CenterGrid object, not {type(grid)}")

        self._grid = grid

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grid={self.grid})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with {self.grid}"

    @property
    def grid(self) -> CenterGrid:
        """`.CenterGrid` instance."""
        return self._grid

    @cached_property
    def b_rho(self) -> NDArray[np.float64]:
        """Radial covariant basis :math:`\\mathbf{b}_\\rho`.

        :math:`\\mathbf{b}_\\rho` is defined as follows:

        .. math::

            \\mathbf{b}_\\rho = \\frac{\\partial \\mathbf{r}}{\\partial \\rho}

        where :math:`\\mathbf{r}` is the position vector of the center grids.

        The shape array of :math:`\\mathbf{b}_\\rho` is (L, M, N, 3), which follows the order of
        (:math:`\\rho, \\theta, \\zeta`) grid resolution at the first three dimensions, and the
        last dimension is the coordinate of :math:`(x, y, z)`.
        """
        L, M, N = self.grid.shape

        b_rho = np.zeros((L, M, N, 3), dtype=float)

        # calculate rho covariant basis
        for n, m in np.ndindex(N, M):
            # calculate length of each segment along to rho direction
            length = np.linalg.norm(self.grid[1:, m, n, :] - self.grid[0:-1, m, n, :], axis=1)

            # reconstruct length array to coordinates starting from 0
            coord = np.hstack(([0.0], np.cumsum(length)))

            # calculate rho covariant basis with numpy.gradient
            b_rho[:, m, n, :] = np.gradient(self.grid[:, m, n, :], coord, axis=0)

        return b_rho

    @cached_property
    def b_theta(self) -> NDArray[np.float64]:
        """Poloidal covariant basis :math:`\\mathbf{b}_\\theta`.

        :math:`\\mathbf{b}_\\theta` is defined as follows:

        .. math::

            \\mathbf{b}_\\theta = \\frac{\\partial \\mathbf{r}}{\\partial \\theta}

        where :math:`\\mathbf{r}` is the position vector of the center grids.

        The shape array of :math:`\\mathbf{b}_\\rho` is (L, M, N, 3), which follows the order of
        (:math:`\\rho, \\theta, \\zeta`) grid resolution at the first three dimensions, and the
        last dimension is the coordinate of :math:`(x, y, z)`.
        """
        L, M, N = self.grid.shape

        b_theta = np.zeros((L, M, N, 3), dtype=float)

        # calculate theta covariant basis
        for l, n in np.ndindex(L, N):
            # calculate length of each segment along to theta direction
            length = np.linalg.norm(self.grid[l, 1:, n, :] - self.grid[l, 0:-1, n, :], axis=1)

            # reconstruct length array to coordinates starting from 0
            coord = np.hstack(([0.0], np.cumsum(length)))

            # calculate theta covariant basis with numpy.gradient
            b_theta[l, :, n, :] = np.gradient(self.grid[l, :, n, :], coord, axis=0)

        return b_theta

    @cached_property
    def b_zeta(self) -> NDArray[np.float64]:
        """Toroidal covariant basis :math:`\\mathbf{b}_\\zeta`.

        :math:`\\mathbf{b}_\\zeta` is defined as follows:

        .. math::

            \\mathbf{b}_\\zeta = \\frac{\\partial \\mathbf{r}}{\\partial \\zeta}

        where :math:`\\mathbf{r}` is the position vector of the center grids.

        The shape array of :math:`\\mathbf{b}_\\rho` is (L, M, N, 3), which follows the order of
        (:math:`\\rho, \\theta, \\zeta`) grid resolution at the first three dimensions, and the
        last dimension is the coordinate of :math:`(x, y, z)`.
        """
        L, M, N = self.grid.shape

        b_zeta = np.zeros((L, M, N, 3), dtype=float)

        # calculate zeta covariant basis
        for l, m in np.ndindex(L, M):
            # calculate length of each segment along to zeta direction
            length = np.linalg.norm(self.grid[l, m, 1:, :] - self.grid[l, m, 0:-1, :], axis=1)

            # reconstruct length array to coordinates starting from 0
            coord = np.hstack(([0.0], np.cumsum(length)))

            # calculate zeta covariant basis with numpy.gradient
            b_zeta[l, m, :, :] = np.gradient(self.grid[l, m, :, :], coord, axis=0)

        return b_zeta

    @cached_property
    def jacobian(self) -> NDArray[np.float64]:
        """Jacobian determinant of EMC3-EIRENE-defined center grids.

        Jacobian determinant :math:`J` is calculated with triple product of covariant bases:

        .. math::

            J = \\mathbf{b}_\\rho \\cdot (\\mathbf{b}_\\theta \\times \\mathbf{b}_\\zeta)

        The shape array of :math:`J` is (L, M, N), which follows the order of
        (:math:`\\rho, \\theta, \\zeta`) grid resolution.
        """
        return np.linalg.det(
            np.concatenate(
                (
                    self.b_rho[..., np.newaxis, :],
                    self.b_theta[..., np.newaxis, :],
                    self.b_zeta[..., np.newaxis, :],
                ),
                axis=-2,
            )
        )

    @cached_property
    def b_sup_rho(self) -> NDArray[np.float64]:
        """Radial contravariant basis :math:`\\mathbf{b}^\\rho`.

        :math:`\\mathbf{b}^\\rho` is defined as follows:

        .. math::

            \\mathbf{b}^\\rho = \\frac{\\mathbf{b}_\\theta \\times \\mathbf{b}_\\zeta}{J}

        where :math:`J` is the Jacobian determinant of EMC3-EIRENE-defined center grids.

        The shape array of :math:`\\mathbf{b}^\\rho` is (L, M, N, 3), which follows the order of
        (:math:`\\rho, \\theta, \\zeta`) grid resolution at the first three dimensions, and the
        last dimension is the coordinate of :math:`(x, y, z)`.
        """
        return np.cross(self.b_theta, self.b_zeta, axis=-1) / self.jacobian[..., np.newaxis]

    @cached_property
    def b_sup_theta(self) -> NDArray[np.float64]:
        """Poloidal contravariant basis :math:`\\mathbf{b}^\\theta`.

        :math:`\\mathbf{b}^\\theta` is defined as follows:

        .. math::

            \\mathbf{b}^\\theta = \\frac{\\mathbf{b}_\\zeta \\times \\mathbf{b}_\\rho}{J}

        where :math:`J` is the Jacobian determinant of EMC3-EIRENE-defined center grids.

        The shape array of :math:`\\mathbf{b}^\\theta` is (L, M, N, 3), which follows the order of
        (:math:`\\rho, \\theta, \\zeta`) grid resolution at the first three dimensions, and the
        last dimension is the coordinate of :math:`(x, y, z)`.
        """
        return np.cross(self.b_zeta, self.b_rho, axis=-1) / self.jacobian[..., np.newaxis]

    @cached_property
    def b_sup_zeta(self) -> NDArray[np.float64]:
        """Toroidal contravariant basis :math:`\\mathbf{b}^\\zeta`.

        :math:`\\mathbf{b}^\\zeta` is defined as follows:

        .. math::

            \\mathbf{b}^\\zeta = \\frac{\\mathbf{b}_\\rho \\times \\mathbf{b}_\\theta}{J}

        where :math:`J` is the Jacobian determinant of EMC3-EIRENE-defined center grids.

        The shape array of :math:`\\mathbf{b}^\\zeta` is (L, M, N, 3), which follows the order of
        (:math:`\\rho, \\theta, \\zeta`) grid resolution at the first three dimensions, and the
        last dimension is the coordinate of :math:`(x, y, z)`.
        """
        return np.cross(self.b_rho, self.b_theta, axis=-1) / self.jacobian[..., np.newaxis]

    def compute_metric(
        self, v1: NDArray[np.float64], v2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute metric tensor.

        The metric tensor :math:`g_{ij}` or :math:`g^{ij}` is calculated with inner product of
        covariant or contravariant bases:

        .. math::

            \\begin{aligned}
                g_{ij} &= \\mathbf{b}_i \\cdot \\mathbf{b}_j \\\\
                g^{ij} &= \\mathbf{b}^i \\cdot \\mathbf{b}^j
            \\end{aligned}

        where :math:`\\mathbf{b}^i` is the contravariant basis of :math:`i`-th coordinate, and
        :math:`\\mathbf{b}_j` is the covariant basis of :math:`j`-th coordinate.

        Parameters
        ----------
        v1 : array_like
            :math:`i`-th co/contra-variant basis.
        v2 : array_like
            :math:`j`-th co/contra-variant basis.

        Returns
        -------
        array_like
            Metric tensor :math:`g_{ij}` or :math:`g^{ij}`.

        Examples
        --------
        >>> import numpy as np
        >>> from cherab.lhd.emc3 import CenterGrid

        >>> grid = CenterGrid("zone0", index_type="coarse")
        >>> coords = CurvCoords(grid)

        Let compute the metric temsor: :math:`\\mathbf{b}_\\rho \\cdot \\mathbf{b}^\\rho`.
        It should be an array of ones.

        >>> g1 = coords.compute_metric(coords.b_rho, coords.b_sup_rho)
        >>> g1.shape
        (33, 100, 9)
        >>> np.allclose(g1, np.ones_like(g1))
        True

        Let check if the metric tensor: :math:`\\mathbf{b}_\\rho\\cdot\\mathbf{b}_\\rho` is not
        the same as the before.

        >>> g2 = coords.compute_metric(coords.b_rho, coords.b_rho)
        >>> np.allclose(g1, g2)
        False
        >>> g2[0, 0, 0]
        """

        return np.einsum("...i,...i", v1, v2)
