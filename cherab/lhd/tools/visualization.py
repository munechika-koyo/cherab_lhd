"""Module relating to visualizing, plotting, etc."""
from __future__ import annotations

from collections.abc import Callable, Collection
from multiprocessing import Manager, Process, Queue, cpu_count
from numbers import Real

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import AsinhNorm, LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import (
    AsinhLocator,
    AutoLocator,
    AutoMinorLocator,
    EngFormatter,
    LogFormatterSciNotation,
    LogLocator,
    MultipleLocator,
    PercentFormatter,
    ScalarFormatter,
    SymmetricalLogLocator,
)
from mpl_toolkits.axes_grid1.axes_grid import CbarAxesBase, ImageGrid

from cherab.core.math import PolygonMask2D, sample2d

from ..machine import wall_outline
from .samplers import sample3d_rz

__all__ = [
    "show_profile_phi_degs",
    "show_profiles_rz_plane",
    "set_axis_properties",
    "set_norm",
    "set_cbar_format",
]


# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6

# Default phi values
PHIS = np.linspace(0, 17.99, 6)


def show_profile_phi_degs(
    func: Callable[[float, float, float], Real],
    fig: Figure | None = None,
    phi_degs: Collection[float] = PHIS,
    nrows_ncols: tuple[int, int] | None = None,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    resolution: float = 5.0e-3,
    mask: str | None = "<=0",
    vmax: float | None = None,
    vmin: float | None = 0.0,
    clabel: str | None = None,
    cmap: str = "plasma",
    plot_mode: str = "scalar",
    cbar_format: str | None = None,
    linear_width: float = 1.0,
    show_phi: bool = True,
    **kwargs,
) -> tuple[Figure, ImageGrid]:
    """
    show EMC3-EIRENE discretized data function in r - z plane with several toroidal angles.

    Parameters
    ----------
    func
        callable object
    fig
        Figure object, by default `plt.figure()`
    phi_degs
        toroidal angles, by default `np.linspace(0, 17.99, 6)`
    nrows_ncols
        Number of rows and columns in the grid, by default None.
        If None, this is automatically rearranged by the length of `.phi_degs`.
    rz_range
        sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``
    resolution
        sampling resolution, by default 5.0e-3
    mask
        masking profile by the following method:
        ``"wall"`` - profile is masked in the wall outline LHD.
        ``"grid"`` - profile is masked in the EMC3-EIRENE-defined grid if `func` has `inside_grid`
        attributes.
        ``"<0"`` - profile is masked less than zero values.
        ``"<=0"`` - profile is masked bellow zero values.
        Otherwise (including None) profile is not masked, by default `"<=0"`
    vmax
        maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values
    vmin
        minimum value of colorbar limits, by default 0.0.
        If None, minimum value is chosen of all sampled values
    clabel
        colorbar label, by default None
    cmap
        colorbar map, by default "plasma"
    plot_mode
        the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``, ``"symlog"``, ``"asinh"``},
        by default ``"scalar"``.
        Each mode corresponds to the :obj:`~matplotlib.colors.Normalize` object as follows.
    cbar_format
        formatter for colorbar yaxis major locator, by default None.
        If None, the formatter is automatically set by `plot_mode`.
    linear_width
        linear width of asinh/symlog norm, by default 1.0
    show_phi
        If True, toroidal angle is annotated in each axis, by default True
    **kwargs : :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        *kwargs* are used to specify properties,
        by default `axes_pad=0.0`, `label_mode="L"`, `cbar_mode="single"`.

    Returns
    -------
    tuple[:obj:`~matplotlib.figure.Figure`, :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`]
    """
    if nrows_ncols is not None:
        if not (isinstance(nrows_ncols, tuple) and len(nrows_ncols) == 2):
            raise TypeError("nrows_ncols must be list containing two elements.")
        nrows = nrows_ncols[0]
        ncols = nrows_ncols[1]
        if nrows < 1:
            nrows = round(len(phi_degs) / ncols)
        elif ncols < 1:
            ncols = round(len(phi_degs) / nrows)

        if nrows * ncols < len(phi_degs):
            raise ValueError("nrows_ncols must have numbers over length of phi_degs.")

        nrows_ncols = (nrows, ncols)

    else:
        nrows_ncols = (1, len(phi_degs))

    # set default cbar_format
    if cbar_format is None:
        cbar_format = plot_mode

    # sampling rate
    rmin, rmax, zmin, zmax = rz_range
    if rmin >= rmax or zmin >= zmax:
        raise ValueError("Invalid rz_range.")

    nr = round((rmax - rmin) / resolution)
    nz = round((zmax - zmin) / resolution)

    # figure object
    if not isinstance(fig, Figure):
        fig = plt.figure()

    # set default ImageGrid parameters
    grid_params = dict(**kwargs)
    grid_params.setdefault("axes_pad", 0.0)
    grid_params.setdefault("label_mode", "L")
    grid_params.setdefault("cbar_mode", "single")

    grids = ImageGrid(fig, 111, nrows_ncols, **grid_params)

    # === parallelized sampling ====================================================================
    manager = Manager()
    profiles = manager.dict()
    job_queue = manager.Queue()

    # create tasks
    for i, phi_deg in enumerate(phi_degs):
        job_queue.put((i, phi_deg))

    # produce worker pool
    pool_size = min(len(phi_degs), cpu_count())
    workers = [
        Process(
            target=_worker1,
            args=(func, mask, (rmin, rmax, nr), (zmin, zmax, nz), job_queue, profiles),
        )
        for _ in range(pool_size)
    ]
    for p in workers:
        p.start()

    for p in workers:
        p.join()

    # ==============================================================================================

    # maximum value of all profiles
    data_max = np.amax([profile.max() for profile in profiles.values()])
    data_min = np.amin([profile.min() for profile in profiles.values()])

    # validate vmax
    if vmax is None:
        vmax = data_max
    if vmin is None:
        vmin = data_min

    # set norm
    norm = set_norm(plot_mode, vmin, vmax, linear_width=linear_width)

    # r, z grids
    r_pts = np.linspace(rmin, rmax, nr)
    z_pts = np.linspace(zmin, zmax, nz)

    for i, phi_deg in enumerate(phi_degs):
        # mapping
        grids[i].pcolormesh(r_pts, z_pts, profiles[i], cmap=cmap, shading="auto", norm=norm)

        # annotation of toroidal angle
        if show_phi:
            grids[i].text(
                rmin + (rmax - rmin) * 0.02,
                zmax - (zmax - zmin) * 0.02,
                f"$\\phi=${phi_deg:.1f}$^\\circ$",
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
            )

        # set each axis properties
        set_axis_properties(grids[i])

    # set colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    extend = _set_cbar_extend(vmin, vmax, data_min, data_max)
    cbar = plt.colorbar(mappable, grids.cbar_axes[0], extend=extend)

    # set colorbar label
    cbar.set_label(clabel)

    # set colorbar's locator and formatter
    set_cbar_format(cbar.ax, cbar_format, linear_width=linear_width)

    cbar_text = cbar.ax.yaxis.get_offset_text()
    x, y = cbar_text.get_position()
    cbar_text.set_position((x * 3.0, y))

    # set axis labels
    nrow, ncol = grids.get_geometry()
    for i in range(nrow):
        grids[i * ncol].set_ylabel("$Z$ [m]")
    for i in range(ncol):
        grids[i + (nrow - 1) * ncol].set_xlabel("$R$ [m]")

    return (fig, grids)


def show_profiles_rz_plane(
    funcs: list[Callable[[float, float, float], Real]],
    fig: Figure | None = None,
    phi_deg: float = 0.0,
    mask: str | None = "<=0",
    nrows_ncols: tuple[int, int] | None = None,
    labels: list[str] | None = None,
    vmax: float | None = None,
    vmin: float | None = 0.0,
    resolution: float = 5.0e-3,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    clabels: list[str] | str = "",
    cmap: str = "plasma",
    cbar_mode: str = "single",
    plot_mode: str = "scalar",
    cbar_format: str | None = None,
    linear_width: float = 1.0,
    **kwargs,
) -> tuple[Figure, ImageGrid]:
    """
    show several EMC3-EIRENE discretized data functions in one r - z plane.

    Parameters
    ----------
    funcs
        each callable object is a different E3E profile function
    fig
        Figure object, by default `plt.figure()`
    phi_deg
        toroidal angle, by default 0.0
    nrows_ncols
        Number of rows and columns in the grid, by default None.
        If None, this is automatically rearranged by the length of `.funcs`.
    mask
        masking profile by the following method:
        ``"wall"`` - profile is masked in the wall outline LHD.
        ``"grid"`` - profile is masked in the EMC3-EIRENE-defined grid if `func` has `inside_grid`
        attributes.
        ``"<0"`` - profile is masked less than zero values.
        ``"<=0"`` - profile is masked bellow zero values.
        Otherwise (including None) profile is not masked, by default ``"<=0"``
    labels
        each profile title is renderered in each axis, by default None.
    vmax
        maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values
    vmin
        minimum value of colorbar limits, by default 0.0.
        If None, minimum value is chosen of all sampled values
    resolution
        sampling resolution, by default 5.0e-3
    rz_range
        sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``
    clabels
        list of colorbar labels, by default "".
        If the length of clabels is less than the length of funcs, the last element of clabels is
        used for all colorbars when cbar_mode is "single".
    cmap
        colorbar map, by default "plasma"
    cbar_mode
        ImgeGrid's parameter to set colorbars in ``"single"`` axes or ``"each"`` axes,
        by default ``"single"``
    plot_mode
        the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``, ``"symlog"``, ``"asinh"``},
        by default ``"scalar"``.
        Each mode corresponds to the :obj:`~matplotlib.colors.Normalize` object as follows.
    cbar_format
        formatter for colorbar yaxis major locator, by default None.
        If None, the formatter is automatically set by `plot_mode`.
    linear_width
        linear width of asinh/symlog norm, by default 1.0
    **kwargs : :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        *kwargs* are used to specify properties, by default axes_pad=0.0, label_mode="L".

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

    # check clabels
    if isinstance(clabels, str):
        clabels = [clabels for _ in range(len(funcs))]
    elif isinstance(clabels, list):
        if len(clabels) < len(funcs):
            raise ValueError("The length of clabels must be equal to or greater than funcs.")
    else:
        raise TypeError("clabels must be str or list.")

    # set default cbar_format
    if cbar_format is None:
        cbar_format = plot_mode

    # sampling rate
    rmin, rmax, zmin, zmax = rz_range
    if rmin >= rmax or zmin >= zmax:
        raise ValueError("Invalid rz_range.")

    nr = round((rmax - rmin) / resolution)
    nz = round((zmax - zmin) / resolution)

    # figure object
    if not isinstance(fig, Figure):
        fig = plt.figure()

    # set default ImageGrid parameters
    grid_params = dict(**kwargs)
    grid_params.setdefault("axes_pad", 0.0)
    grid_params.setdefault("label_mode", "L")

    # Initiate ImageGrid
    grids = ImageGrid(fig, 111, nrows_ncols, cbar_mode=cbar_mode, **grid_params)

    # === parallelized sampling ====================================================================
    manager = Manager()
    profiles = manager.dict()
    job_queue = manager.Queue()

    # create tasks
    for i, func in enumerate(funcs):
        job_queue.put((i, func))

    # produce worker pool
    pool_size = min(len(funcs), cpu_count())
    workers = [
        Process(
            target=_worker2,
            args=(phi_deg, mask, (rmin, rmax, nr), (zmin, zmax, nz), job_queue, profiles),
        )
        for _ in range(pool_size)
    ]
    for p in workers:
        p.start()

    for p in workers:
        p.join()

    # === display image ============================================================================

    # get maximum and minimum value of each profile
    _vmaxs = [profile.max() for profile in profiles.values()]
    _vmins = [profile.min() for profile in profiles.values()]

    # define vmaxs
    if isinstance(vmax, (float, int)):
        vmaxs: list[float] = [vmax for _ in range(len(profiles))]
    else:
        vmaxs: list[float] = _vmaxs

    # define vmins
    if isinstance(vmin, (float, int)):
        vmins: list[float] = [vmin for _ in range(len(profiles))]
    else:
        vmins: list[float] = _vmins

    if cbar_mode == "single":
        vmaxs: list[float] = [max(vmaxs) for _ in range(len(vmaxs))]
        vmins: list[float] = [min(vmins) for _ in range(len(vmins))]

    # r, z grids
    r_pts = np.linspace(rmin, rmax, nr)
    z_pts = np.linspace(zmin, zmax, nz)

    for i in range(len(profiles)):
        # mapping
        mappable = grids[i].pcolormesh(
            r_pts, z_pts, profiles[i], cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
        )

        # annotation of toroidal angle
        if isinstance(labels, Collection) and len(labels) >= len(profiles):
            grids[i].text(
                rmin + (rmax - rmin) * 0.02,
                zmin - (zmax - zmin) * 0.02,
                f"{labels[i]}",
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="square, pad=0.1", edgecolor="k", facecolor="w", linewidth=0.8),
            )

        # set each axis properties
        set_axis_properties(grids[i])

    # create colorbar objects and store them into a list
    cbars = []
    if cbar_mode == "each":
        for i, grid in enumerate(grids):
            extend = _set_cbar_extend(vmins[i], vmaxs[i], _vmins[i], _vmaxs[i])
            cbar = plt.colorbar(grid.images[0], grids.cbar_axes[i], extend=extend)
            cbars.append(cbar)

    else:  # cbar_mode == "single"
        vmax, vmin = max(vmaxs), min(vmins)
        _vmax, _vmin = max(_vmaxs), min(_vmins)
        extend = _set_cbar_extend(vmin, vmax, _vmin, _vmax)
        norm = set_norm(plot_mode, vmins[0], vmaxs[0], linear_width=linear_width)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable, grids.cbar_axes[0], extend=extend)
        cbars.append(cbar)

    # set colorbar's locator and formatter
    for cbar, clabel in zip(cbars, clabels, strict=True):
        set_cbar_format(cbar.ax, cbar_format, linear_width=linear_width)
        cbar.set_label(clabel)

    # set axis labels
    nrow, ncol = grids.get_geometry()
    for i in range(nrow):
        grids[i * ncol].set_ylabel("$Z$ [m]")
    for i in range(ncol):
        grids[i + (nrow - 1) * ncol].set_xlabel("$R$ [m]")

    return (fig, grids)


def _worker1(
    func: Callable[[float, float, float], Real],
    mask: str | None,
    r_range: tuple[float, float, int],
    z_range: tuple[float, float, int],
    job_queue: Queue,
    profiles: dict,
) -> None:
    """Worker process to generate sampled & masked profiles."""
    while not job_queue.empty():
        try:
            # extract a task
            index, phi_deg = job_queue.get(block=False)

            # generate profile
            profile = _sampler(func, phi_deg, mask, r_range, z_range)

            profiles[index] = profile

        except Exception:
            break


def _worker2(
    phi_deg: float,
    mask: str | None,
    r_range: tuple[float, float, int],
    z_range: tuple[float, float, int],
    job_queue: Queue,
    profiles: dict,
) -> None:
    """Worker process to generate sampled & masked profiles."""
    while not job_queue.empty():
        try:
            # extract a task
            index, func = job_queue.get(block=False)

            # generate profile
            profile = _sampler(func, phi_deg, mask, r_range, z_range)

            profiles[index] = profile

        except Exception:
            break


def _sampler(
    func: Callable[[float, float, float], Real],
    phi_deg: float,
    mask: str | None,
    r_range: tuple[float, float, int],
    z_range: tuple[float, float, int],
) -> np.ndarray:
    """Sampler for function at any toroidal angle."""
    # sampling
    _, _, sampled = sample3d_rz(func, r_range, z_range, phi_deg)

    # generate mask array
    # TODO: use np.ma.masked_where
    match mask:
        case "wall":
            wall_contour = wall_outline(phi_deg, basis="rz")
            inside_wall = PolygonMask2D(wall_contour[:-1, :].copy(order="C"))
            _, _, mask_arr = sample2d(inside_wall, r_range, z_range)
            mask_arr = np.logical_not(mask_arr)

        case "grid":
            if inside_grids := getattr(func, "inside_grids", None):
                _, _, mask_arr = sample3d_rz(inside_grids, r_range, z_range, phi_deg)
                mask_arr = np.logical_not(mask_arr)
            else:
                mask_arr = sampled < 0

        case "<0":
            mask_arr = sampled < 0

        case "<=0":
            mask_arr = sampled <= 0

        case _:
            mask_arr = np.zeros_like(sampled, dtype=bool)

    # generate masked sampled array
    profile: np.ndarray = np.transpose(np.ma.masked_array(sampled, mask=mask_arr))

    return profile


def set_axis_properties(axes: Axes) -> Axes:
    """Set x-, y-axis property. This function set axis labels and tickers.

    Parameters
    ----------
    axes
        matplotlib Axes object

    Returns
    -------
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


def set_norm(mode: str, vmin: float, vmax: float, linear_width: float = 1.0) -> Normalize:
    """Set variouse :obj:`~matplotlib.colors.Normalize` object.

    Parameters
    ----------
    mode
        the way of normalize the data scale.
        Must select one in {``"scalar"``, ``"log"``, ``"centered"``, ``"symlog"``, ``"asinh"``}
    vmin
        minimum value of the profile.
    vmax
        maximum value of the profile.
    linear_width
        linear width of asinh/symlog norm, by default 1.0

    Returns
    -------
    Normalize
        norm object
    """
    # set norm
    absolute = max(abs(vmax), abs(vmin))
    match mode:
        case "log":
            if vmin <= 0:
                raise ValueError("vmin must be positive value.")
            norm = LogNorm(vmin=vmin, vmax=vmax)

        case "symlog":
            norm = SymLogNorm(linthresh=linear_width, vmin=-1 * absolute, vmax=absolute)

        case "centered":
            norm = Normalize(vmin=-1 * absolute, vmax=absolute)

        case "asinh":
            norm = AsinhNorm(linear_width=linear_width, vmin=-1 * absolute, vmax=absolute)

        case _:
            norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


def set_cbar_format(cax: CbarAxesBase | Axes, formatter: str, linear_width: float = 1.0, **kwargs):
    """Set colorbar's locator and formatter.

    Parameters
    ----------
    cax
        colorbar axes object
    formatter
        formatter for colorbar yaxis major locator.
        Must select one in {``"scalar"``, ``"log"``, ``"symlog"``, ``"asinh"``, ``percent``, ``eng``}
    linear_width
        linear width of asinh/symlog norm, by default 1.0
    **kwargs
        keyword arguments for formatter

    Returns
    -------
    Colorbar
        colorbar object with new properties
    """
    # define colobar formatter and locator
    match formatter:
        case "log":
            fmt = LogFormatterSciNotation(**kwargs)
            major_locator = LogLocator(base=10, numticks=None)
            minor_locator = LogLocator(base=10, subs=tuple(np.arange(0.1, 1.0, 0.1)), numticks=12)

        case "symlog":
            fmt = LogFormatterSciNotation(linthresh=linear_width, **kwargs)
            major_locator = SymmetricalLogLocator(linthresh=linear_width, base=10)
            minor_locator = SymmetricalLogLocator(
                linthresh=linear_width, base=10, subs=tuple(np.arange(0.1, 1.0, 0.1))
            )

        case "asinh":
            fmt = LogFormatterSciNotation(linthresh=linear_width, **kwargs)
            major_locator = AsinhLocator(linear_width=linear_width, base=10)
            minor_locator = AsinhLocator(
                linear_width=linear_width, base=10, subs=tuple(np.arange(0.1, 1.0, 0.1))
            )

        case "percent":
            fmt = PercentFormatter(**kwargs)
            major_locator = AutoLocator()
            minor_locator = AutoMinorLocator()

        case "eng":
            fmt = EngFormatter(**kwargs)
            major_locator = AutoLocator()
            minor_locator = AutoMinorLocator()

        case _:
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            major_locator = AutoLocator()
            minor_locator = AutoMinorLocator()

    # set colorbar's locator and formatter
    cax.yaxis.set_offset_position("left")
    cax.yaxis.set_major_formatter(fmt)
    cax.yaxis.set_major_locator(major_locator)
    cax.yaxis.set_minor_locator(minor_locator)


def _set_cbar_extend(user_vmin: float, user_vmax: float, data_vmin: float, data_vmax: float) -> str:
    """Set colorbar's extend.

    Parameters
    ----------
    user_vmin
        user defined minimum value.
    user_vmax
        user defined maximum value.
    data_vmin
        minimum value of the profile.
    data_vmax
        maximum value of the profile.

    Returns
    -------
    str
        colorbar's extend.
    """
    if data_vmin < user_vmin:
        if user_vmax < data_vmax:
            extend = "both"
        else:
            extend = "min"
    else:
        if user_vmax < data_vmax:
            extend = "max"
        else:
            extend = "neither"

    return extend


if __name__ == "__main__":
    pass
