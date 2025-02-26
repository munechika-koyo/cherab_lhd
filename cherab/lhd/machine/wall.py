"""Module to offer wall contour features."""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from numpy import float64, intp
from numpy.typing import NDArray

from ..tools.fetch import fetch_file

__all__ = [
    "wall_outline",
    "plot_lhd_wall_outline",
    "periodic_toroidal_angle",
    "adjacent_toroidal_angles",
]


def periodic_toroidal_angle(phi: float) -> tuple[float, bool]:
    """Return toroidal angle & z coordinate under periodic boundary condition.

    The specified toroidal angle by EMC3-EIRENE varies from 0 to 18 degrees. For example,
    the poloidal grid plane at 27 degrees corresponds to the one flipped along the z-axis at
    9 degrees.

    Parameters
    ----------
    phi : float
        Toroidal angle in degree.

    Returns
    -------
    phi : float
        Toroidal angle in degree between 0 and 18.
    flipped : bool
        Flag of flipping z coordinate.
        If `flipped` is ``True``, :math:`z` component is multiplied by -1.
    """
    if phi < 0.0:
        phi = (phi + 360.0) % 36.0
    else:
        phi %= 36.0

    if phi < 18.0:
        flipped = False
    else:
        phi = 36.0 - phi
        flipped = True
    return (phi, flipped)


def adjacent_toroidal_angles(phi: float, phis: np.ndarray) -> tuple[intp, intp]:
    """Generate adjacent toroidal angles.

    If ``phis = [0.0, 0.5, 1.0,..., 18.0]`` and given ``phi = 0.75``, then (left, right) adjacent
    toroidal angles are (0.5, 1.0), each index of which is (1, 2), respectively.

    Parameters
    ----------
    phi : float
        Toroidal angle between 0 and 18 degree.
    phis : (N, ) array_like
        1D array of toroidal angles.

    Returns
    -------
    tuple[int, int]
        (left, right) adjacent toroidal angle indices.
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
    """:math:`(r, z)` or :math:`(x, y, z)` coordinates of LHD wall outline at a toroidal angle
    :math:`\\varphi`.

    If no :math:`(r, z)` coordinates data is at :math:`\\varphi`, then one point of wall outline
    :math:`xyz` is interpolated linearly according to the following equation:

    .. math::

        xyz = \\frac{(\\varphi - \\varphi_i) xyz_{i+1} + (\\varphi_{i+1} - \\varphi) xyz_{i}}{\\varphi_{i+1} - \\varphi_{i}}

    where :math:`\\varphi_{i} < \\varphi < \\varphi_{i+1}` and :math:`xyz_{i}` and :math:`xyz_{i+1}`
    is wall outline coordinates at :math:`\\varphi_{i}` and :math:`\\varphi_{i+1}`, respectively.

    Parameters
    ----------
    phi : float
        Toroidal angle in units of degree.
    basis : {"rz", "xyz"}, optional
        Coordinate system for returned points, by default "rz".

    Returns
    -------
    (N, 2) or (N, 3) array_like
        Wall outline points in either :math:`(r, z)` or :math:`(x, y, z)` coordinates which depends
        on the `basis` parameter.

    Examples
    --------
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

    # Load wall outline dataset
    path = fetch_file("machine/wall_outline.nc")
    da = xr.open_dataarray(path)
    phis = da["ζ"].values
    outlines = da.values

    # phi -> phi in 0 - 18 deg
    phi_t, flipped = periodic_toroidal_angle(phi)

    # find adjacent phis
    phi_left, phi_right = adjacent_toroidal_angles(phi_t, phis)

    # load rz wall outline
    rz_left = outlines[phi_left, :, :]
    rz_right = outlines[phi_right, :, :]

    # flipped value for z axis
    flip = -1 if flipped else 1

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
    """Plot LHD vessel wall polygons in a :math:`r` - :math:`z` plane.

    Parameters
    ----------
    phi : float
        Toroidal angle in unit of degree.

    Examples
    --------
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
