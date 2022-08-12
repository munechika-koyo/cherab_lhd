from collections.abc import Callable
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, AutoLocator, ScalarFormatter

from cherab.core.math import PolygonMask2D
from cherab.core.math import sample2d
from cherab.lhd.machine import wall_outline
from cherab.lhd.tools import sample3d_rz

from multiprocessing import Process, Manager


# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6


def show_profile_phi_degs(
    func: Callable[[float, float, float], int | float],
    phi_degs=np.linspace(0, 17.99, 6),
    nrows_ncols: tuple[int, int] = None,
    resolution=5.0e-3,
    masked: str = "wall",
    max_value: float = None,
    clabel: str = None,
    cmap="plasma",
    **kwargs,
) -> tuple[plt.Figure, ImageGrid]:
    """
    show E3E discretized data function in r - z plane with several toroidal angles.

    Parameters
    ----------
    func : callable
        callable object
    phi_degs : list[float], optional
        toroidal angles, by default `np.linspace(0, 17.99, 6)`
    nrows_ncols : tuple[int, int], optional
        Number of rows and columns in the grid, by default None.
        If None, this is automatically rearranged by the length of `.phi_degs`.
    resolution : float, optional
        sampling resolution, by default 5.0e-3
    masked : str, optional
        masking profile by the following method:
        If ``"wall"``, profile is masked in the wall outline LHD.
        If ``"EMC3"``, profile is masked in the EMC3 grid outline which derived from :obj:`.EMC3Mask`(:obj:`func`).
        If ``"bellow_zero"``, profile is masked bellow zero values.
        Otherwise profile is not masked, by default "wall"
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

    # sampling rate
    nr = int(round((RMAX - RMIN) / resolution))
    nz = int(round((ZMAX - ZMIN) / resolution))

    fig = plt.figure()

    # set default ImageGrid parameters
    grid_params = dict(**kwargs)
    grid_params.setdefault("axes_pad", 0.0)
    grid_params.setdefault("label_mode", "L")
    grid_params.setdefault("cbar_mode", "single")

    grids = ImageGrid(fig, 111, nrows_ncols, **grid_params)

    # parallelized sampling
    manager = Manager()
    profiles_dict: dict
    profiles_dict = manager.dict()

    for i, phi_deg in enumerate(phi_degs):
        process = Process(
            target=_sampling_function,
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
        mappable = grids[i].pcolormesh(
            r_pts, z_pts, profiles_dict[i], cmap=cmap, shading="auto", vmin=0, vmax=max_value
        )

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
    funcs: list[Callable[[float, float, float], int | float]],
    phi_deg=0.0,
    masked="wall",
    nrows_ncols: tuple[int, int] = None,
    labels: list[str] = [],
    max_value: float = None,
    resolution=5.0e-3,
    clabel: str = None,
    cmap="plasma",
    **kwargs,
) -> tuple[plt.Figure, ImageGrid]:
    """
    show several E3E discretized data functions in one r - z plane.

    Parameters
    ----------
    funcs : list of callable
        each callable object is a different E3E profile function
    phi_deg : float, optional
        toroidal angle, by default 0.0
    nrows_ncols : (int, int)
        Number of rows and columns in the grid, by default None.
        If None, this is automatically rearranged by the length of `.funcs`.
    masked : str, optional
        masking profile by the following method:
        If ``"wall"``, profile is masked in the wall outline LHD.
        If ``"EMC3"``, profile is masked in the EMC3 grid outline which derived from :obj:`.EMC3Mask`(:obj:`func`).
        If ``"bellow_zero"``, profile is masked bellow zero values.
        Otherwise profile is not masked, by default "wall"
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

    # sampling rate
    nr = int(round((RMAX - RMIN) / resolution))
    nz = int(round((ZMAX - ZMIN) / resolution))

    fig = plt.figure()

    # set default ImageGrid parameters
    grid_params = dict(**kwargs)
    grid_params.setdefault("axes_pad", 0.0)
    grid_params.setdefault("label_mode", "L")
    grid_params.setdefault("cbar_mode", "single")

    grids = ImageGrid(fig, 111, nrows_ncols, **grid_params)

    # parallelized sampling
    manager = Manager()
    profiles_dict: dict
    profiles_dict = manager.dict()

    for i, func in enumerate(funcs):
        process = Process(
            target=_sampling_function,
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
        mappable = grids[i].pcolormesh(
            r_pts, z_pts, profiles_dict[i], cmap=cmap, shading="auto", vmin=0, vmax=max_value
        )

        # annotation of toroidal angle
        if len(labels) > i:
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


def _sampling_function(phi_deg, func, masked, nr, nz, profiles_dict, process_index):
    """excute one sampling process.
    This is defined for multiple processing
    """

    # sampling
    _, _, fvar = sample3d_rz(func, (RMIN, RMAX, nr), (ZMIN, ZMAX, nz), phi_deg)

    # sampling masked function
    if masked == "wall":
        wall_contour, _ = wall_outline(phi_deg)
        inside_wall = PolygonMask2D(wall_contour[:-1, :].copy(order="C"))
        _, _, mask = sample2d(inside_wall, (RMIN, RMAX, nr), (ZMIN, ZMAX, nz))

    elif masked == "EMC3":
        _, _, mask = sample3d_rz(func.inside_grids, (RMIN, RMAX, nr), (ZMIN, ZMAX, nz), phi_deg)

    elif masked == "bellow_zero":
        mask = fvar > 0

    else:
        mask = np.ones_like(fvar, dtype=np.bool8)

    # masking
    profile = np.transpose(np.ma.masked_array(fvar, mask=np.logical_not(mask)))

    # store
    profiles_dict[process_index] = profile


def set_axis_properties(axes: plt.Axes) -> plt.Axes:
    """
    Set x-, y-axis property.
    This function set axis labels and tickers.

    Parameter
    ---------
    axes : :obj:`~matplotlib.axes.Axes`

    Return
    ------
    :obj:`~matplotlib.axes.Axes`
        axes object with new properties
    """
    axes.set_xlabel("R[m]")
    axes.xaxis.set_minor_locator(MultipleLocator(0.1))
    axes.yaxis.set_minor_locator(MultipleLocator(0.1))
    axes.xaxis.set_major_formatter("{x:.1f}")
    axes.yaxis.set_major_formatter("{x:.1f}")
    axes.tick_params(direction="in", labelsize=10, which="both", top=True, right=True)

    return axes


if __name__ == "__main__":
    from cherab.lhd.emitter.E3E import EMC3, EMC3Mapper, DataLoader

    index_func = EMC3.load_index_func()
    loader = DataLoader()
    radiation = EMC3Mapper(index_func, loader.radiation())
    fig, grids = show_profiles_rz_plane(
        [radiation], masked=None, phi_deg=0.0, clabel=r"$P_{rad}$ [W/m$^3$]"
    )
    fig.show()
