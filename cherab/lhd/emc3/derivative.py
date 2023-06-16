"""Contains functions for derivative matrix computation.

Here we assume that the mesh configuration is based on a original EMC3-EIRENE one. (2023-06-13)
"""
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from .barycenters import EMC3CenterGrids

__all__ = ["compute_derivative_matrix", "partial_derivatives", "jacobian_matrix"]


def compute_derivative_matrix(grids: EMC3CenterGrids, direction: str = "x") -> NDArray[np.float64]:
    """Compute the derivative matrix of the EMC3-EIRENE barycenter grids.

    The derivative matrix is computed with the central difference method.
    The partial derivatives of the EMC3-EIRENE barycenter grids are computed with
    :func:`partial_derivatives`.

    .. todo::

        Consider the procedure of the total zone connections.
        For now only one zone can be handled.

    Parameters
    ----------
    grids
        The EMC3-EIRENE Barycenter grids.
    direction
        Name of the direction to compute the derivative matrix.
        It can be ``"x"``, ``"y"`` or ``"z"``, by default ``"x"``.

    Returns
    -------
    NDArray[np.float64]
        The derivative matrix of the EMC3-EIRENE grids.
    """
    # get the shape of the grids
    L, M, N = grids.shape

    # initialize the derivative matrix
    dmat = lil_matrix((L * M * N, L * M * N))

    for n in range(N):
        for m in range(M):
            for l in range(L):
                # compute the jacobian matrix
                jmat = jacobian_matrix(l, m, n, grids)

                # compute partial derivatives
                p_rho, p_theta, p_zeta = partial_derivatives(l, m, n, jmat, var=direction)
                index = l + m * L + n * L * M

                # partial derivatives of rho
                if l == 0:
                    index0 = index
                    index1 = index0 + 1
                    index2 = index0 + 2
                    dmat[index, index0] = -1.5 * p_rho
                    dmat[index, index1] = 2.0 * p_rho
                    dmat[index, index2] = -0.5 * p_rho

                elif l == L - 1:
                    # assuming zero outside the boundary
                    index_b = index - 1
                    dmat[index, index_b] = -0.5 * p_rho

                else:
                    index_f = index + 1
                    index_b = index - 1
                    dmat[index, index_f] = 0.5 * p_rho
                    dmat[index, index_b] = -0.5 * p_rho

                # partial derivatives of theta
                # NOTE: Considering the periodic boundary condition for zone0, zone11
                if m == 0:
                    index_f = l + (m + 1) * L + n * L * M
                    index_b = l + (M - 1) * L + n * L * M
                    dmat[index, index_f] = 0.5 * p_theta
                    dmat[index, index_b] = -0.5 * p_theta

                elif m == M - 1:
                    index_f = l + 0 * L + n * L * M
                    index_b = l + (m - 1) * L + n * L * M
                    dmat[index, index_f] = 0.5 * p_theta
                    dmat[index, index_b] = -0.5 * p_theta

                else:
                    index_f = l + (m + 1) * L + n * L * M
                    index_b = l + (m - 1) * L + n * L * M
                    dmat[index, index_f] = 0.5 * p_theta
                    dmat[index, index_b] = -0.5 * p_theta

                # partial derivatives of zeta
                if n == 0:
                    index0 = index
                    index1 = l + m * L + (n + 1) * L * M
                    index2 = l + m * L + (n + 2) * L * M
                    dmat[index, index0] = -1.5 * p_zeta
                    dmat[index, index1] = 2.0 * p_zeta
                    dmat[index, index2] = -0.5 * p_zeta

                elif n == N - 1:
                    index0 = index
                    index1 = l + m * L + (n - 1) * L * M
                    index2 = l + m * L + (n - 2) * L * M
                    dmat[index, index0] = 1.5 * p_zeta
                    dmat[index, index1] = -2.0 * p_zeta
                    dmat[index, index2] = 0.5 * p_zeta

                else:
                    index_f = l + m * L + (n + 1) * L * M
                    index_b = l + m * L + (n - 1) * L * M
                    dmat[index, index_f] = 0.5 * p_zeta
                    dmat[index, index_b] = -0.5 * p_zeta

    return dmat


def partial_derivatives(
    l: int, m: int, n: int, jacobian_mat: NDArray[np.float64], var: str = "x"
) -> tuple[float, float, float]:
    """Compute the partial derivatives of the EMC3-EIRENE grids :math:`\\rho, \\theta, \\zeta`.

    Each barycenter is located at the center of the cell and indexed by ``(l, m, n)``.
    ``l``, ``m`` and ``n`` are the indices of the cell in the radial :math:`\\rho`,
    poloidal :math:`\\theta` and toroidal-like :math:`\\zeta` directions, respectively.
    They follow the curvilinear coordinates system of the EMC3-EIRENE mesh.

    For example, a partial derivative of :math:`\\partial \\rho / \\partial x \\equiv \\rho_x` is
    computed as follows:

    .. math::

        \\rho_x = \\frac{1}{\\mathrm{det} J} \\left(y_\\theta z_\\zeta - y_\\zeta z_\\theta\\right).

    :math:`J` is the jacobian matrix of the EMC3-EIRENE grids, and :math:`\\mathrm{det} J`
    is its determinant.

    Each partial derivative at :math:`(x, y, z)` indexed by ``(l, m, n)`` is computed with the
    central difference method like:

    .. math::

        x_\\rho \\approx \\frac{x_{l+1, m, n} - x_{l-1, m, n}}{2}.


    Parameters
    ----------
    l
        Index of the cell in the radial direction :math:`\\rho`.
    m
        Index of the cell in the poloidal direction :math:`\\theta`.
    n
        Index of the cell in the toroidal-like direction :math:`\\zeta`.
    jacobian_mat
        The 3x3 jacobian matrix of the EMC3-EIRENE grids.
    var
        Name of the variable to compute the partial derivatives.
        It can be ``"x"``, ``"y"`` or ``"z"``, by default ``"x"``.

    Returns
    -------
    tuple[float, float, float]
        The partial derivatives like (:math:`\\rho_x, \\theta_x, \\zeta_x`) of the EMC3-EIRENE barycenters.
    """
    jacobian = np.linalg.det(jacobian_mat)

    if var == "x":
        p_rho = (
            jacobian_mat[1, 1] * jacobian_mat[2, 2] - jacobian_mat[1, 2] * jacobian_mat[2, 1]
        ) / jacobian
        p_theta = (
            jacobian_mat[1, 2] * jacobian_mat[2, 0] - jacobian_mat[1, 0] * jacobian_mat[2, 2]
        ) / jacobian
        p_zeta = (
            jacobian_mat[1, 0] * jacobian_mat[2, 1] - jacobian_mat[1, 1] * jacobian_mat[2, 0]
        ) / jacobian

    elif var == "y":
        p_rho = (
            jacobian_mat[0, 2] * jacobian_mat[2, 1] - jacobian_mat[0, 1] * jacobian_mat[2, 2]
        ) / jacobian
        p_theta = (
            jacobian_mat[0, 0] * jacobian_mat[2, 2] - jacobian_mat[0, 2] * jacobian_mat[2, 0]
        ) / jacobian
        p_zeta = (
            jacobian_mat[0, 1] * jacobian_mat[2, 0] - jacobian_mat[0, 0] * jacobian_mat[2, 1]
        ) / jacobian

    elif var == "z":
        p_rho = (
            jacobian_mat[0, 1] * jacobian_mat[1, 2] - jacobian_mat[0, 2] * jacobian_mat[1, 1]
        ) / jacobian
        p_theta = (
            jacobian_mat[0, 2] * jacobian_mat[1, 0] - jacobian_mat[0, 0] * jacobian_mat[1, 2]
        ) / jacobian
        p_zeta = (
            jacobian_mat[0, 0] * jacobian_mat[1, 1] - jacobian_mat[0, 1] * jacobian_mat[1, 0]
        ) / jacobian

    else:
        raise ValueError("Invalid variable name. Please use 'x', 'y' or 'z'.")

    return (p_rho, p_theta, p_zeta)


def jacobian_matrix(l: int, m: int, n: int, grids: EMC3CenterGrids) -> NDArray[np.float64]:
    """Compute the jacobian matrix of the EMC3-EIRENE grids.

    Each barycenter is located at the center of the cell and indexed by ``(l, m, n)``.
    ``l``, ``m`` and ``n`` are the indices of the cell in the radial :math:`\\rho`,
    poloidal :math:`\\theta` and toroidal-like :math:`\\zeta` directions, respectively.
    They follow the curvilinear coordinates system of the EMC3-EIRENE mesh, so the jacobian
    matrix :math:`J` is defined as follows:

    .. math::

        J =
        \\begin{bmatrix}
            x_\\rho & x_\\theta & x_\\zeta\\\\
            y_\\rho & y_\\theta & y_\\zeta\\\\
            z_\\rho & z_\\theta & z_\\zeta
        \\end{bmatrix},

    where :math:`x_\\rho` represents :math:`\\frac{\\partial x}{\\partial \\rho}`, and so on.

    :math:`J` is computed at the barycenter of the cell ``(l, m, n)``.

    Each partial derivative at :math:`(x, y, z)` indexed by ``(l, m, n)`` is computed with the
    central difference method like:

    .. math::

        x_\\rho \\approx \\frac{x_{l+1, m, n} - x_{l-1, m, n}}{2}.


    At the boundary, the partial derivative is computed with the forward or backward difference
    method like:

    .. math::

        x_\\rho \\approx \\frac{-3 x_{l, m, n} + 4 x_{l+1, m, n} - x_{l+2, m, n}}{2},

    or

    .. math::

        x_\\rho \\approx \\frac{x_{l-2, m, n} - 4 x_{l-1, m, n} + 3 x_{l, m, n}}{2}.


    Parameters
    ----------
    l
        Index of the cell in the radial direction :math:`\\rho`.
    m
        Index of the cell in the poloidal direction :math:`\\theta`.
    n
        Index of the cell in the toroidal-like direction :math:`\\zeta`.
    grids
        The barycenters of the EMC3-EIRENE cell.

    Returns
    -------
    NDArray[np.float64]
        The jacobian matrix :math:`J` of the cell ``(l, m, n)``.
    """
    # get the shape of the grids
    L, M, N = grids.shape

    # define the jacobian matrix
    jacobian = np.zeros((3, 3), dtype=float)

    # compute the jacobian matrix
    # x_rho, y_rho, z_rho
    # TODO: Consider zone-dependent boundary condition.
    # TODO: Consider the connnection to the next zone.
    if grids.zone in {"zone0", "zone11"}:
        if l == 0:
            jacobian[:, 0] = 0.5 * (
                -3.0 * grids.index(l, m, n)
                + 4.0 * grids.index(l + 1, m, n)
                - grids.index(l + 2, m, n)
            )
        elif l == L - 1:
            jacobian[:, 0] = 0.5 * (
                grids.index(l - 2, m, n)
                - 4.0 * grids.index(l - 1, m, n)
                + 3.0 * grids.index(l, m, n)
            )
        else:
            jacobian[:, 0] = 0.5 * (grids.index(l + 1, m, n) - grids.index(l - 1, m, n))
    else:
        raise NotImplementedError("Not implemented yet.")

    # x_theta, y_theta, z_theta
    if grids.zone in {"zone0", "zone11"}:
        if m == 0:
            jacobian[:, 1] = 0.5 * (grids.index(l, m + 1, n) - grids.index(l, M - 1, n))
        elif m == M - 1:
            jacobian[:, 1] = 0.5 * (grids.index(l, 0, n) - grids.index(l, m - 1, n))
        else:
            jacobian[:, 1] = 0.5 * (grids.index(l, m + 1, n) - grids.index(l, m - 1, n))
    else:
        if m == 0:
            jacobian[:, 1] = 0.5 * (
                -3.0 * grids.index(l, m, n)
                + 4.0 * grids.index(l, m + 1, n)
                - grids.index(l, m + 2, n)
            )
        elif m == M - 1:
            jacobian[:, 1] = 0.5 * (
                grids.index(l, m - 2, n)
                - 4.0 * grids.index(l, m - 1, n)
                + 3.0 * grids.index(l, m, n)
            )
        else:
            jacobian[:, 1] = 0.5 * (grids.index(l, m + 1, n) - grids.index(l, m - 1, n))

    # x_zeta, y_zeta, z_zeta
    # TODO: Consider interface between zone0 and zone11.
    # TODO: Consider helical symmetry at 0 and 18 degree.
    if n == 0:
        jacobian[:, 2] = 0.5 * (
            -3.0 * grids.index(l, m, n) + 4.0 * grids.index(l, m, n + 1) - grids.index(l, m, n + 2)
        )
    elif n == N - 1:
        jacobian[:, 2] = 0.5 * (
            grids.index(l, m, n - 2) - 4.0 * grids.index(l, m, n - 1) + 3.0 * grids.index(l, m, n)
        )
    else:
        jacobian[:, 2] = 0.5 * (grids.index(l, m, n + 1) - grids.index(l, m, n - 1))

    return jacobian


if __name__ == "__main__":
    grids = EMC3CenterGrids("zone0")
    print(jacobian_matrix(1, 1, 1, grids))
