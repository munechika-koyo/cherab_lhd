"""Module to offer wall contour fetures."""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy import float64, intp
from numpy.typing import NDArray

__all__ = [
    "wall_outline",
    "plot_lhd_wall_outline",
    "periodic_toroidal_angle",
    "adjacent_toroidal_angles",
]


DIR_WALL = Path(__file__).parent.resolve() / "geometry" / "data" / "wall_outline"


def periodic_toroidal_angle(phi: float) -> tuple[float, bool]:
    """Return toroidal angle & z coordinate under periodic boundary condition.
    The defined toroidal angle by EMC3-EIRENE varies from 0 to 18 degree. For
    example, the poloidal grid plane in 27 degree corresponds to fliped one in
    9 degree.

    Parameters
    ----------
    phi
        toroidal angle

    Returns
    -------
    tuple[float, bool]
        (toroidal angle, flag of flipping)
        If this flag is  ``True``, :math:`z` component is multiplied by -1.
    """
    index = phi // 36

    if (phi // 18) % 2 == 0:
        fliped = False
        if index == 0:
            pass
        else:
            phi = phi - 36 * index
    else:
        phi = 36 * (index + 1) - phi
        fliped = True
    return (phi, fliped)


def adjacent_toroidal_angles(phi: float, phis: np.ndarray) -> tuple[intp, intp]:
    """
    Generate adjacent toroidal angles.
    if ``phis = [0.0, 0.5, 1.0,..., 18.0]`` and given ``phi = 0.75``,
    then (left, right) adjacent toroidal angle are (0.5, 1.0), each index of which is (1, 2), respectively.

    Parameters
    ----------
    phi
        toroidal angle betwenn 0 and 18 degree
    phis
        1D array of toroidal angles

    Returns
    -------
    tuple[int, int]
        (left, right) adjacent toroidal angle indices
    """
    if phi < 0.0 or phi > 18.0:
        raise ValueError("phi must be an angle between 0 to 18 degree.")

    index = np.abs(phis - phi).argmin()

    phi_pre = phis[index - 1]
    if index + 1 < phis.size:
        phi_ad = phis[index + 1]

        if abs(phi - phi_pre) < abs(phi - phi_ad):
            return (index - 1, index)
        else:
            return (index, index + 1)
    else:
        return (index - 1, index)


def wall_outline(phi: float, basis: str = "rz") -> NDArray[float64]:
    """
    :math:`(r, z)` or :math:`(x, y, z)` coordinates of LHD wall outline at a toroidal angle :math:`\\varphi`.
    If no :math:`(r, z)` coordinates data is at :math:`\\varphi`,
    then one point of wall outline :math:`xyz` is interpolated linearly according to the following equation:

    .. math::

        xyz = \\frac{(\\varphi - \\varphi_i) xyz_{i+1} + (\\varphi_{i+1} - \\varphi) xyz_{i}}{\\varphi_{i+1} - \\varphi_{i}}

    where :math:`\\varphi_{i} < \\varphi < \\varphi_{i+1}` and :math:`xyz_{i}` and :math:`xyz_{i+1}` is wall outline coordinates at
    :math:`\\varphi_{i}` and :math:`\\varphi_{i+1}`, respectively.

    Parameters
    ----------
    phi
        toroidal angle in units of degree.
    basis : str `{"rz" or "xyz"}`
        coordinate system for returned points, by default `"rz"`

    Returns
    -------
    :obj:`~numpy.ndarray`
        wall outline points in either :math:`(r, z)` or :math:`(x, y, z)` coordinates which depends on
        the `basis` parameter.
        The shape of ndarray is either `(N, 2)` in :math:`(r, z)` or `(N, 3)` in :math:`(x, y, z)`.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from cherab.lhd.machine import wall_outline
        >>> rz = wall_outline(15.0, basis="rz")
        >>> rz
        array([[ 4.40406713,  1.51311291],
               [ 4.39645296,  1.42485631],
               ...
               [ 4.40406713,  1.51311291]])
    """

    # validate basis parameter
    if basis not in {"rz", "xyz"}:
        raise ValueError("basis parameter must be chosen from 'rz' or 'xyz'.}")

    # extract toroidal angles from file name
    filenames = sorted(DIR_WALL.glob("*.txt"))
    phis = np.array([float(file.stem) for file in filenames])

    # phi -> phi in 0 - 18 deg
    phi_t, fliped = periodic_toroidal_angle(phi)

    # find adjacent phis
    phi_left, phi_right = adjacent_toroidal_angles(phi_t, phis)

    # load rz wall outline
    rz_left = np.loadtxt(DIR_WALL / filenames[phi_left]) * 1.0e-2  # [cm] -> [m]
    rz_right = np.loadtxt(DIR_WALL / filenames[phi_right]) * 1.0e-2

    # fliped value for z axis
    flip = -1 if fliped else 1

    xyz_left = np.array(
        [
            rz_left[:, 0] * np.cos(np.deg2rad(phis[phi_left])),
            rz_left[:, 0] * np.sin(np.deg2rad(phis[phi_left])),
            rz_left[:, 1] * flip,
        ]
    )
    xyz_right = np.array(
        [
            rz_right[:, 0] * np.cos(np.deg2rad(phis[phi_right])),
            rz_right[:, 0] * np.sin(np.deg2rad(phis[phi_right])),
            rz_right[:, 1] * flip,
        ]
    )

    # linearly interpolate wall outline
    xyz = ((phi_t - phis[phi_left]) * xyz_right + (phis[phi_right] - phi_t) * xyz_left) / (
        phis[phi_right] - phis[phi_left]
    )

    if basis == "xyz":
        return xyz.T
    else:
        return np.array([np.hypot(xyz[0, :], xyz[1, :]), xyz[2, :]]).T


def plot_lhd_wall_outline(phi: float) -> None:
    """plot LHD vessel wall polygons in a :math:`r` - :math:`z` plane

    Parameters
    ----------
    phi
        toroidal angle in unit of degree

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from cherab.lhd.machine import plot_lhd_wall_outline
        >>> plot_lhd_wall_outline(15.0)

    .. image:: ../_static/images/plotting/plot_lhd_wall_outline.png
    """
    rz = wall_outline(phi, basis="rz")
    plt.plot(rz[:, 0], rz[:, 1])
    plt.xlabel("$R$[m]")
    plt.ylabel("$Z$[m]")
    plt.axis("equal")
    plt.title(f"$\\varphi = ${phi:.1f} deg")


if __name__ == "__main__":
    plot_lhd_wall_outline(15.0)
    plt.show()
    pass
