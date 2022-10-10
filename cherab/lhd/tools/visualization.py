"""
Module relating to visualizing, plotting, etc.
"""
from __future__ import annotations

from collections.abc import Callable, Collection
from multiprocessing import Manager, Process, Queue, cpu_count
from numbers import Real

import numpy as np
from cherab.core.math import PolygonMask2D, sample2d
from cherab.lhd.machine import wall_outline
from cherab.lhd.tools import sample3d_rz
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MultipleLocator, ScalarFormatter
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

__all__ = ["show_profile_phi_degs", "show_profiles_rz_plane", "set_axis_properties"]


# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6


def show_profile_phi_degs(
    func: Callable[[float, float, float], Real],
    fig: Figure | None = None,
    phi_degs: Collection[float] = np.linspace(0, 17.99, 6),
    nrows_ncols: tuple[int, int] | None = None,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    resolution: float = 5.0e-3,
    mask: str | None = "<=0",
    vmax: float | None = None,
    vmin: float = 0.0,
    clabel: str | None = None,
    cmap: str = "plasma",
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
    clabel
        colorbar label, by default None
    cmap
        colorbar map, by default "plasma"
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
        if nrows_ncols[0] * nrows_ncols[1] < len(phi_degs):
            raise ValueError("nrows_ncols must have numbers over length of phi_degs.")
    else:
        nrows_ncols = (1, len(phi_degs))

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
    profiles: dict
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
    temp_max = np.max([profile.max() for profile in profiles.values()])
    min_value = np.min([profile.min() for profile in profiles.values()])

    if vmax is None:
        vmax = temp_max
        extend = "neither"
    elif vmax < temp_max:
        extend = "max"
    elif vmax < 0.0:
        raise ValueError("vmax must be greater than 0.0")
    else:
        extend = "neither"

    # validate vmin
    if not isinstance(vmin, Real):
        raise TypeError("vmin must be a float.")
    if vmin >= vmax:
        raise ValueError(f"vmin: {vmin:d} must be less than vmax: {vmax:d}.")
    if vmin > min_value:
        if extend == "max":
            extend = "both"
        else:
            extend = "min"

    # r, z grids
    r_pts = np.linspace(rmin, rmax, nr)
    z_pts = np.linspace(zmin, zmax, nz)

    for i, phi_deg in enumerate(phi_degs):

        # mapping
        mappable = grids[i].pcolormesh(
            r_pts, z_pts, profiles[i], cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
        )

        # annotation of toroidal angle
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
    funcs: list[Callable[[float, float, float], Real]],
    fig: Figure | None = None,
    phi_deg: float = 0.0,
    mask: str | None = "<=0",
    nrows_ncols: tuple[int, int] | None = None,
    labels: list[str] = [],
    vmax: float | None = None,
    vmin: float = 0.0,
    resolution: float = 5.0e-3,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    clabel: str | None = None,
    cmap: str = "plasma",
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
        each profile title is renderered in each axis.
    vmax
        maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values
    vmin
        minimum value of colorbar limits, by default 0.0.
    resolution
        sampling resolution, by default 5.0e-3
    rz_range
        sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``
    clabel
        colorbar label, by default None
    cmap
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
    profiles: dict
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

    # ==============================================================================================

    # maximum value of all profiles
    temp_max = np.max([profile.max() for profile in profiles.values()])
    min_value = np.min([profile.min() for profile in profiles.values()])

    if vmax is None:
        vmax = temp_max
        extend = "neither"
    elif vmax < temp_max:
        extend = "max"
    elif vmax < 0.0:
        raise ValueError("vmax must be greater than 0.0")
    else:
        extend = "neither"

    # validate vmin
    if not isinstance(vmin, Real):
        raise TypeError("vmin must be a float.")
    if vmin >= vmax:
        raise ValueError(f"vmin: {vmin:d} must be less than vmax: {vmax:d}.")
    if vmin > min_value:
        if extend == "max":
            extend = "both"
        else:
            extend = "min"

    # r, z grids
    r_pts = np.linspace(rmin, rmax, nr)
    z_pts = np.linspace(zmin, zmax, nz)

    for i in range(len(profiles)):

        # mapping
        mappable = grids[i].pcolormesh(
            r_pts, z_pts, profiles[i], cmap=cmap, shading="auto", vmin=vmin, vmax=vmax
        )

        # annotation of toroidal angle
        if len(labels) > i:
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


def _worker1(
    func: Callable[[float, float, float], Real],
    mask: str | None,
    r_range: tuple[float, float, int],
    z_range: tuple[float, float, int],
    job_queue: Queue,
    profiles: dict,
) -> None:
    """worker process to generate sampled & masked profiles"""
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
    """worker process to generate sampled & masked profiles"""
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
    """sampler for function at any toroidal angle."""
    # sampling
    _, _, sampled = sample3d_rz(func, r_range, z_range, phi_deg)

    # generate masked array
    if mask == "wall":
        wall_contour = wall_outline(phi_deg, basis="rz")
        inside_wall = PolygonMask2D(wall_contour[:-1, :].copy(order="C"))
        _, _, mask_arr = sample2d(inside_wall, r_range, z_range)
        mask_arr = np.logical_not(mask_arr)

    elif mask == "grid":
        if hasattr(func, "inside_grids"):
            _, _, mask_arr = sample3d_rz(getattr(func, "inside_grids"), r_range, z_range, phi_deg)
            mask_arr = np.logical_not(mask_arr)
        else:
            mask_arr = sampled < 0

    elif mask == "<0":
        mask_arr = sampled < 0

    elif mask == "<=0":
        mask_arr = sampled <= 0

    else:
        mask_arr = np.zeros_like(sampled, dtype=np.bool8)

    # generate masked sampled array
    profile: np.ndarray = np.transpose(np.ma.masked_array(sampled, mask=mask_arr))

    return profile


def set_axis_properties(axes: Axes) -> Axes:
    """
    Set x-, y-axis property.
    This function set axis labels and tickers.

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


if __name__ == "__main__":
    from cherab.lhd.emc3 import EMC3Mapper, PhysIndex
    from cherab.lhd.emc3.dataio import DataLoader

    print("Instantiating PhysIndex...")
    index_func = PhysIndex()
    print("Loading radiation data...")
    loader = DataLoader()
    radiation = EMC3Mapper(index_func, loader.radiation())
    print("Plotting radiation profile...")
    fig, grids = show_profiles_rz_plane(
        [radiation], mask=None, phi_deg=0.0, clabel=r"$P_{rad}$ [W/m$^3$]"
    )
    fig.show()
    pass
