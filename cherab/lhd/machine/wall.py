import os
import numpy as np
from matplotlib import pyplot as plt


DIR_WALL = os.path.join(os.path.dirname(__file__), "geometry", "data", "wall_outline")


def periodic_toroidal_angle(phi):
    """
    Return toroidal angle & z coordinate under periodic boundary condition.
    The defined toroidal angle by EMC3-EIRENE varies from 0 to 18 degree.
    For example, the poloidal grid plane in 27 degree corresponds to fliped one in 9 degree.

    Parameters
    ----------
    phi : float
        toroidal angle

    Returns
    -------
    tuple of (float, bool)
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


def adjacent_toroidal_angles(phi, phis):
    """
    Generate adjacent toroidal angles.
    if ``phis = [0.0, 0.5, 1.0,..., 18.0]`` and given ``phi = 0.75``,
    then (left, right) adjacent toroidal angle are (0.5, 1.0), each index of which is (1, 2), respectively.

    Parameters
    ----------
    phi : float
        toroidal angle betwenn 0 and 18 degree
    phis : :obj:`~numpy.ndarray`
        1D array of toroidal angles

    Returns
    -------
    tuple
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


def wall_outline(phi):
    """
    :math:`(r, z)` or :math:`(x, y, z)` coordinates of LHD wall outline at a toroidal angle :math:`\\varphi`.
    If no :math:`(r, z)` coordinates data is at :math:`\\varphi`,
    then one point of wall outline :math:`xyz` is interpolated linearly according to the following equation:

    .. math::

        xyz = \\frac{(\\varphi - \\varphi_i) xyz_{i+1} + (\\varphi_{i+1} - \\varphi) xyz_{i}}{\\varphi_{i+1} - \\varphi_{i}}

    where :math:`\\varphi_{i} < \\varphi < \\varphi_{i+1}` and :math:`xyz_{i}` and :math:`xyz_{i+1}` is wall outline coordinates at
    :math:`\\varphi_{i}` and :math:`\\varphi_{i+1}`, respectively.

    Parameter
    ---------
    phi : float
        toroidal angle in units of degree.

    Return
    ------
    tuple of (:obj:`~numpy.ndarray`, :obj:`~numpy.ndarray`)
        each of which is a wall outline in :math:`(r, z)` and :math:`(x, y, z)` coordinates, respectively
    """

    # extract toroidal angles from file name
    filenames = os.listdir(DIR_WALL)
    filenames.sort()
    phis = np.array([float(file.strip(".txt")) for file in filenames])

    # phi -> phi in 0 - 18 deg
    phi_t, fliped = periodic_toroidal_angle(phi)

    # find adjacent phis
    phi_left, phi_right = adjacent_toroidal_angles(phi_t, phis)

    # load rz wall outline
    rz_left = np.loadtxt(os.path.join(DIR_WALL, filenames[phi_left])) * 1.0e-3  # [cm] -> [m]
    rz_right = np.loadtxt(os.path.join(DIR_WALL, filenames[phi_right])) * 1.0e-3

    # fliped value for z axis
    flip = -1 if fliped else 1

    xyz_left = np.array(
        [
            rz_left[:, 0] * np.cos(np.deg2rad(phis[phi_left])),
            rz_left[:, 0] * np.sin(np.deg2rad(phis[phi_left])),
            rz_left[:, 1] * flip
        ]
    )
    xyz_right = np.array(
        [
            rz_right[:, 0] * np.cos(np.deg2rad(phis[phi_right])),
            rz_right[:, 0] * np.sin(np.deg2rad(phis[phi_right])),
            rz_right[:, 1] * flip
        ]
    )

    # linearly interpolate wall outline
    xyz = ((phi_t - phis[phi_left]) * xyz_right + (phis[phi_right] - phi_t) * xyz_left) / (phis[phi_right] - phis[phi_left])

    rz = np.array(
        [
            np.hypot(xyz[0, :], xyz[1, :]),
            xyz[2, :]
        ]
    )

    return (rz, xyz)


def plot_lhd_wall_outline(phi):
    """plot LHD vessel wall polygons

    Parameters
    ----------
    phi : float
        toroidal angle in unit of degree
    """
    rz, _ = wall_outline(phi)
    plt.plot(rz[0, :], rz[1, :])
    plt.xlabel("$R$[m]")
    plt.ylabel("$Z$[m]")
    plt.axis("equal")
    plt.title(f"$\\varphi = ${phi:.1f} deg")


if __name__ == "__main__":
    plot_lhd_wall_outline(15.0)
    plt.show()
    pass
