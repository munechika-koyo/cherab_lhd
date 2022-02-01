from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, AutoLocator, ScalarFormatter

from cherab.lhd.emitter.E3E import Discrete3DMesh, EMC3Mapper
from cherab.lhd.tools import sample3d_rz

from multiprocessing import Process, Manager

# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6


def show_profile_phi_degs(
    func, phi_degs=np.linspace(0, 17.99, 6), nrows_ncols=(2, 3), resolution=5.0e-3, masked=False, max_value=None, clabel=None, cmap="plasma", **kwargs
):
    """
    show E3E discretized data function in r - z plane with several toroidal angles.

    Parameters
    ----------
    func : callable
        callable object
    phi_degs : list of float, optional
        toroidal angles, by default `np.linspace(0, 17.99, 6)`
    nrows_ncols : (int, int)
        Number of rows and columns in the grid, by default (2, 3).
        If None, this is automatically rearranged by the length of `.phi_degs`.
    resolution : float, optional
        sampling resolution, by default 5.0e-3
    masked : bool or callable, optional
        if True, sampled values are masked using func's :obj:`~EMC3Mapper.inside_grids`.
        if callable is given, sampled values are masked by the callable.
        callable must inherit :obj:`~.Discrete3DMesh`, by default False
    max_value : float, optional
        maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values
    clabel : str, optional
        colorbar label, by default None
    cmap: str, optional
        colorbar map, by default "plasma"
    **kwargs : :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        *kwargs* are used to specify properties, by default axes_pad=0.0, label_mode="L", cbar_mode="single".

    Returns
    -------
    tuple of :obj:`~matplotlib.figure.Figure`, :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`
    """

    if nrows_ncols:
        if not (isinstance(nrows_ncols, tuple) and len(nrows_ncols) == 2):
            raise TypeError("nrows_ncols must be list containing two elements.")
        if nrows_ncols[0] * nrows_ncols[1] < len(phi_degs):
            raise ValueError("nrows_ncols must have numbers over length of phi_degs.")
    else:
        nrows_ncols = (1, len(phi_degs))

    if masked:
        if callable(masked):
            if not isinstance(masked, Discrete3DMesh):
                raise TypeError("callable masked must be a index function which returns -1 ouside undefined-grid area")
        else:
            if not isinstance(func, EMC3Mapper):
                raise TypeError("func must be instantiated by EMC3Mapper if masked == True.")
            masked = func.inside_grids

    # sampling rate
    nr = int(round((RMAX - RMIN) / resolution))
    nz = int(round((ZMAX - ZMIN) / resolution))

    fig = plt.figure()

    # set default ImageGrid parameters
    grid_params = defaultdict(**kwargs)
    grid_params.setdefault("axes_pad", 0.0)
    grid_params.setdefault("label_mode", "L")
    grid_params.setdefault("cbar_mode", "single")

    grids = ImageGrid(fig, 111, nrows_ncols, **grid_params)

    # parallelized sampling
    manager = Manager()
    profiles_dict = manager.dict()

    for i, phi_deg in enumerate(phi_degs):
        process = Process(
            target=_sampling_funcion,
            kwargs={
                "phi_deg": phi_deg,
                "func": func,
                "masked": masked,
                "nr": nr,
                "nz": nz,
                "profiles_dict": profiles_dict,
                "process_index": i,
            },
        )
        process.start()

    process.join()

    # maximum value of all profiles
    temp_max = np.max([profile.max() for profile in profiles_dict.values()])
    if not max_value:
        max_value = temp_max
        extend = "neither"
    elif max_value < temp_max:
        extend = "max"
    elif max_value < 0.0:
        raise ValueError("max_value must be greater than 0.0")
    else:
        extend = "neither"

    # r, z grids
    r_pts = np.linspace(RMIN, RMAX, nr)
    z_pts = np.linspace(ZMIN, ZMAX, nz)

    for i, phi_deg in enumerate(phi_degs):

        # mapping
        mappable = grids[i].pcolormesh(r_pts, z_pts, profiles_dict[i], cmap=cmap, shading="auto", vmin=0, vmax=max_value)

        # annotation of toroidal angle
        grids[i].text(
            RMIN + 0.1,
            ZMAX - 0.1,
            f"$\\phi=${phi_deg:.1f}$^\\circ$",
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
        )

        # set each axis
        grids[i].set_xlabel("R[m]")
        grids[i].xaxis.set_minor_locator(MultipleLocator(0.1))
        grids[i].yaxis.set_minor_locator(MultipleLocator(0.1))
        grids[i].xaxis.set_major_formatter("{x:.1f}")
        grids[i].yaxis.set_major_formatter("{x:.1f}")
        grids[i].tick_params(direction="in", labelsize=10, which="both", top=True, right=True)

    # set colorbar
    cbar = plt.colorbar(mappable, grids.cbar_axes[0], extend=extend)
    cbar.set_label(clabel)
    cbar.ax.yaxis.set_major_locator(AutoLocator())
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar_text = cbar.ax.yaxis.get_offset_text()
    x, y = cbar_text.get_position()
    cbar_text.set_position((x * 3.0, y))

    # set yaxis label
    nrow, ncol = grids.get_geometry()
    for i in range(nrow):
        grids[i * ncol].set_ylabel("Z[m]")

    return (fig, grids)


def show_profiles_rz_plane(
    funcs, phi_deg=0.0, masked=False, nrows_ncols=None, labels=None, max_value=None, resolution=5.0e-3, clabel=None, cmap="plasma", **kwargs
):
    """
    show several E3E discretized data functions in one r - z plane.

    Parameters
    ----------
    funcs : list of callable
        each callable object is a different E3E profile function
    phi_deg : float, optional
        toroidal angle, by default 0.0
    nrows_ncols : (int, int)
        Number of rows and columns in the grid.
        This is automatically rearranged by the length of `.funcs`.
    masked : bool or callable, optional
        if True, sampled values masked by :obj:`.EMC3Mask`(:obj:`func`) is shown.
        if callable is given, sampled values masked by the callable is shown.
        callbel must inherit :obj:`~.Discrete3DMesh`, by default False
    labels : list of str, optional
        each profile title is renderered in each axis.
    max_value : float, optional
        maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values
    resolution : float, optional
        sampling resolution, by default 5.0e-3
    clabel : str, optional
        colorbar label, by default None
    cmap: str, optional
        colorbar map, by default "plasma"
    **kwargs : :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        *kwargs* are used to specify properties, by default axes_pad=0.0, label_mode="L", cbar_mode="single".

    Returns
    -------
    tuple of :obj:`~matplotlib.figure.Figure`, :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`
    """
    # validation
    if not isinstance(funcs, list):
        raise TypeError("funcs must be a list type.")

    if nrows_ncols:
        if not (isinstance(nrows_ncols, tuple) and len(nrows_ncols) == 2):
            raise TypeError("nrows_ncols must be list containing two elements.")
        if nrows_ncols[0] * nrows_ncols[1] < len(funcs):
            raise ValueError("nrows_ncols must have numbers over length of funcs.")
    else:
        nrows_ncols = (1, len(funcs))

    if masked:
        if callable(masked):
            if not isinstance(masked, Discrete3DMesh):
                raise TypeError("callable masked must be a index function which returns -1 ouside undefined-grid area")
        else:
            if not isinstance(funcs[0], EMC3Mapper):
                raise TypeError("funcs[0] must be instantiated by EMC3Mapper.")
            masked = funcs[0].inside_grids

    # sampling rate
    nr = int(round((RMAX - RMIN) / resolution))
    nz = int(round((ZMAX - ZMIN) / resolution))

    fig = plt.figure()

    # set default ImageGrid parameters
    grid_params = defaultdict(**kwargs)
    grid_params.setdefault("axes_pad", 0.0)
    grid_params.setdefault("label_mode", "L")
    grid_params.setdefault("cbar_mode", "single")

    grids = ImageGrid(fig, 111, nrows_ncols, **grid_params)

    # parallelized sampling
    manager = Manager()
    profiles_dict = manager.dict()

    for i, func in enumerate(funcs):
        process = Process(
            target=_sampling_funcion,
            kwargs={
                "phi_deg": phi_deg,
                "func": func,
                "masked": masked,
                "nr": nr,
                "nz": nz,
                "profiles_dict": profiles_dict,
                "process_index": i,
            },
        )
        process.start()

    process.join()

    # maximum value of all profiles
    temp_max = np.max([profile.max() for profile in profiles_dict.values()])
    if not max_value:
        max_value = temp_max
        extend = "neither"
    elif max_value < temp_max:
        extend = "max"
    elif max_value < 0.0:
        raise ValueError("max_value must be greater than 0.0")
    else:
        extend = "neither"

    # r, z grids
    r_pts = np.linspace(RMIN, RMAX, nr)
    z_pts = np.linspace(ZMIN, ZMAX, nz)

    for i in range(len(profiles_dict)):

        # mapping
        mappable = grids[i].pcolormesh(r_pts, z_pts, profiles_dict[i], cmap=cmap, shading="auto", vmin=0, vmax=max_value)

        # annotation of toroidal angle
        grids[i].text(
            RMIN + 0.1,
            ZMAX - 0.1,
            f"{labels[i]}",
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
        )

        # set each axis
        grids[i].set_xlabel("R[m]")
        grids[i].xaxis.set_minor_locator(MultipleLocator(0.1))
        grids[i].yaxis.set_minor_locator(MultipleLocator(0.1))
        grids[i].xaxis.set_major_formatter("{x:.1f}")
        grids[i].yaxis.set_major_formatter("{x:.1f}")
        grids[i].tick_params(direction="in", labelsize=10, which="both", top=True, right=True)

    # set colorbar
    cbar = plt.colorbar(mappable, grids.cbar_axes[0], extend=extend)
    cbar.set_label(clabel)
    cbar.ax.yaxis.set_major_locator(AutoLocator())
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar_text = cbar.ax.yaxis.get_offset_text()
    x, y = cbar_text.get_position()
    cbar_text.set_position((x * 3.0, y))

    # set yaxis label
    nrow, ncol = grids.get_geometry()
    for i in range(nrow):
        grids[i * ncol].set_ylabel("Z[m]")

    return (fig, grids)


def _sampling_funcion(phi_deg, func, masked, nr, nz, profiles_dict, process_index):
    """excute one sampling process.
    This is defined for multiple processing
    """

    # sampling
    _, _, fvar = sample3d_rz(func, (RMIN, RMAX, nr), (ZMIN, ZMAX, nz), phi_deg)

    # sampling masked function
    if masked:
        _, _, mask = sample3d_rz(masked, (RMIN, RMAX, nr), (ZMIN, ZMAX, nz), phi_deg)
    else:
        mask = np.ones_like(fvar, dtype=np.bool8)

    # masking
    profile = np.transpose(np.ma.masked_array(fvar, mask=np.logical_not(mask)))

    # store
    profiles_dict[process_index] = profile
