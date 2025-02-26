"""Module relating to visualizing, plotting, etc."""

from __future__ import annotations

from collections.abc import Callable, Collection
from multiprocessing import Manager, Process, Queue, cpu_count
from numbers import Real
from typing import Literal, TypeAlias

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import XAxis, YAxis
from matplotlib.cm import ScalarMappable
from matplotlib.colors import AsinhNorm, Colormap, ListedColormap, LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
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
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

from cherab.core.math import PolygonMask2D, sample2d  # type: ignore

from ..machine import wall_outline
from .samplers import sample3d_rz, sample_xy_plane

__all__ = [
    "show_profile_phi_degs",
    "show_profiles_rz_plane",
    "show_profile_xy_plane",
    "set_axis_properties",
    "set_norm",
    "set_axis_format",
    "add_inner_title",
    "CMAP_RED",
]

# Type aliases
PlotMode: TypeAlias = Literal["scalar", "log", "centered", "symlog", "asinh"]
FormatMode: TypeAlias = Literal["scalar", "log", "symlog", "asinh", "percent", "eng"]
MaskMode: TypeAlias = Literal["wall", "grid", "<0", "<=0"]

# Const.
RMIN = 2.0  # [m]
RMAX = 5.5
ZMIN = -1.6
ZMAX = 1.6

# Default phi values
PHIS = np.linspace(0, 17.99, 6)
PHIS5 = np.linspace(0, 17.99, 5)

# Basis indices
BASIS_INDICES = [0, 1, 2, 3, 4, 5]

# custom Red colormap extracted from "RdBu_r"
cmap = plt.get_cmap("RdBu_r")  # type: ignore
CMAP_RED = ListedColormap(cmap(np.linspace(0.5, 1.0, 256)))


def show_profile_phi_degs(
    func: Callable[[float, float, float], Real],
    fig: Figure | None = None,
    phi_degs: Collection[float] = PHIS,
    nrows_ncols: tuple[int, int] | None = None,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    resolution: float = 5.0e-3,
    mask: MaskMode | None = "grid",
    vmax: float | None = None,
    vmin: float | None = 0.0,
    clabel: str | None = None,
    cmap: str | Colormap = CMAP_RED,
    plot_mode: PlotMode = "scalar",
    cbar_format: FormatMode | PlotMode | None = None,
    linear_width: float | None = None,
    show_phi: bool = True,
    parallel: bool = True,
    **kwargs,
) -> tuple[Figure, ImageGrid]:
    """Show EMC3-EIRENE discretized data function in R-Z plane with several toroidal angles.

    Parameters
    ----------
    func : Callable[[float, float, float], Real]
        Callable object. The function must have three arguments :math:`(X, Y, Z)`.
    fig : `~matplotlib.figure.Figure`, optional
        Figure object, by default `plt.figure()`.
    phi_degs : Collection[float], optional
        Toroidal angles, by default `np.linspace(0, 17.99, 6)`.
    nrows_ncols : tuple[int, int], optional
        Number of rows and columns in the grid, by default None.
        If None, this is automatically rearranged by the length of `.phi_degs`.
    rz_range : tuple[float, float, float, float], optional
        Sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``.
    resolution : float, optional
        Sampling resolution, by default 5.0e-3.
    mask : {"wall", "grid", "<0", "<=0"}, optional
        Masking profile by the following method:
        ``"wall"`` - profile is masked in the wall outline LHD.
        ``"grid"`` - profile is masked in the EMC3-EIRENE-defined grid if `func` has `inside_grid`
        attributes. If not, profile is not masked.
        ``"<0"`` - profile is masked less than zero values.
        ``"<=0"`` - profile is masked below zero values.
        Otherwise (including None) profile is not masked, by default `"grid"`.
    vmax : float, optional
        Maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values.
    vmin : float, optional
        Minimum value of colorbar limits, by default 0.0.
        If None, minimum value is chosen of all sampled values.
    clabel : str, optional
        Colorbar label, by default None.
    cmap : str | `~matplotlib.colors.Colormap`, optional
        Colorbar map, by default `CMAP_RED` (custom Red colormap extracted from "RdBu_r").
    plot_mode : {"scalar", "log", "centered", "symlog", "asinh"}, optional
        Which scale to adapt to the colormap, by default `"scalar"`.
        Each mode corresponds to the `~matplotlib.colors.Normalize` object.
    cbar_format : {"scalar", "log", "symlog", "asinh", "percent", "eng"}, optional
        Formatter to colorbar's major y-axis locator, by default None.
        If None, the formatter is automatically set to the same one determined by `plot_mode`.
    linear_width : float, optional
        Linear width of ``asinh/symlog`` norm, by default None.
        If None, the linear width is automatically set to two digit below the absolute maximum or
        minimum value of the data.
    show_phi : bool, optional
        If True, toroidal angle is annotated in each axis, by default True.
    parallel : bool, optional
        If True, the sampling is parallelized, by default True.
    **kwargs : `~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        User-specified properties, by default `axes_pad=0.0`, `label_mode="L"` and
        `cbar_mode="single"`.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure object.
    grids : `~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`
        ImageGrid object.
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

    # === sampling =================================================================================
    if parallel:
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

    else:
        profiles = {}
        for i, phi_deg in enumerate(phi_degs):
            profiles[i] = _sampler(func, phi_deg, mask, (rmin, rmax, nr), (zmin, zmax, nz))
    # ==============================================================================================

    # maximum value of all profiles
    data_max = np.amax([profile.max() for profile in profiles.values()])
    data_min = np.amin([profile.min() for profile in profiles.values()])

    # validate vmax
    if vmax is None:
        vmax = data_max
    if vmin is None:
        vmin = data_min

    if linear_width is None:
        linear_width = _set_leaner_width(vmin, vmax)

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
            add_inner_title(grids[i], f"$\\phi=${phi_deg:.1f}°")

        # set each axis properties
        set_axis_properties(grids[i])

    # set colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    extend = _set_cbar_extend(vmin, vmax, data_min, data_max)
    cbar = plt.colorbar(mappable, grids.cbar_axes[0], extend=extend)

    # set colorbar label
    cbar.set_label(clabel)

    # set colorbar's locator and formatter
    set_axis_format(cbar.ax.yaxis, cbar_format, linear_width=linear_width)

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
    list_data: list[np.ndarray],
    index_func: Callable[[float, float, float], int],
    fig: Figure | None = None,
    phi_deg: float = 0.0,
    nrows_ncols: tuple[int, int] | None = None,
    mask: MaskMode | None = "grid",
    labels: list[str] | None = None,
    vmax: float | None = None,
    vmin: float | None = 0.0,
    resolution: float = 5.0e-3,
    rz_range: tuple[float, float, float, float] = (RMIN, RMAX, ZMIN, ZMAX),
    clabels: list[str] | str = "",
    cmap: str | Colormap = CMAP_RED,
    cbar_mode: Literal["single", "each"] = "single",
    plot_mode: PlotMode = "scalar",
    cbar_format: FormatMode | PlotMode | None = None,
    linear_width: float | None = None,
    **kwargs,
) -> tuple[Figure, ImageGrid]:
    """Show several EMC3-EIRENE discretized data functions in one R-Z plane.

    Parameters
    ----------
    list_data : list[np.ndarray]
        List of numpy.ndarray data.
    index_func : Callable[[float, float, float], int]
        Callable object to get index of EMC3 meshes.
    fig : `~matplotlib.figure.Figure`, optional
        Figure object, by default `plt.figure()`.
    phi_deg : float, optional
        Toroidal angle, by default 0.0.
    nrows_ncols : tuple[int, int], optional
        Number of rows and columns in the grid, by default None.
        If None, this is automatically rearranged by the length of `.funcs`.
    mask : {"wall", "grid", "<0", "<=0"}, optional
        Masking profile by the following method:
        ``"wall"`` - profile is masked in the wall outline LHD.
        ``"grid"`` - profile is masked in the EMC3-EIRENE-defined grid if `func` has `inside_grid`
        attributes. If not, profile is not masked.
        ``"<0"`` - profile is masked less than zero values.
        ``"<=0"`` - profile is masked below zero values.
        Otherwise (including None) profile is not masked, by default ``"grid"``.
    labels : list[str], optional
        Profile titles written in each axis, by default None.
    vmax : float, optional
        Maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values.
    vmin : float, optional
        Minimum value of colorbar limits, by default 0.0.
        If None, minimum value is chosen of all sampled values.
    resolution : float, optional
        Sampling resolution, by default 5.0e-3.
    rz_range : tuple[float, float, float, float], optional
        Sampling range : :math:`(R_\\text{min}, R_\\text{max}, Z_\\text{min}, Z_\\text{max})`,
        by default ``(2.0, 5.5, -1.6, 1.6)``.
    clabels : list[str] | str, optional
        List of colorbar labels, by default "".
        If the length of clabels is less than the length of funcs, the last element of clabels is
        used for all colorbars when cbar_mode is "single".
    cmap : str | `~matplotlib.colors.Colormap`, optional
        Colorbar map, by default `CMAP_RED` (custom Red colormap extracted from "RdBu_r").
    cbar_mode : {"single", "each"}, optional
        ImgeGrid's parameter to set colorbars in ``"single"`` axes or ``"each"`` axes,
        by default ``"single"``.
    plot_mode : {"scalar", "log", "centered", "symlog", "asinh"}, optional
        Which scale to adapt to the colormap, by default ``"scalar"``.
        Each mode corresponds to the `~matplotlib.colors.Normalize` object.
    cbar_format : {"scalar", "log", "symlog", "asinh", "percent", "eng"}, optional
        Formatter to colorbar's major y-axis locator, by default None.
        If None, the formatter is automatically set to the same one determined by `plot_mode`.
    linear_width : float, optional
        Linear width of ``asinh/symlog`` norm, by default None.
        If None, the linear width is automatically set to two digit below the absolute maximum or
        minimum value of the data.
    **kwargs : :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        User-specified properties, by default `axes_pad=0.0`, `label_mode="L"` and
        `cbar_mode="single"`.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure object.
    grids : `~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`
        ImageGrid object.
    """
    # validation
    if not isinstance(list_data, list):
        raise TypeError("list_data must be a list type.")

    if nrows_ncols:
        if not (isinstance(nrows_ncols, tuple) and len(nrows_ncols) == 2):
            raise TypeError("nrows_ncols must be list containing two elements.")
        if nrows_ncols[0] * nrows_ncols[1] < len(list_data):
            raise ValueError("nrows_ncols must have numbers over length of list_data.")
    else:
        nrows_ncols = (1, len(list_data))

    # check clabels
    if cbar_mode != "single":
        if isinstance(clabels, str):
            clabels = [clabels for _ in range(len(list_data))]
        elif isinstance(clabels, list):
            if len(clabels) < len(list_data):
                raise ValueError(
                    "The length of clabels must be equal to or greater than list_data."
                )
        else:
            raise TypeError("clabels must be str or list.")
    else:
        if isinstance(clabels, str):
            clabels = [clabels]
        else:
            raise TypeError("clabels must be str if cbar_mode is 'single'.")

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
    for i, data in enumerate(list_data):
        job_queue.put((i, data))

    # produce worker pool
    pool_size = min(len(list_data), cpu_count())
    workers = [
        Process(
            target=_worker2,
            args=(
                phi_deg,
                index_func,
                mask,
                (rmin, rmax, nr),
                (zmin, zmax, nz),
                job_queue,
                profiles,
            ),
        )
        for _ in range(pool_size)
    ]
    for p in workers:
        p.start()

    for p in workers:
        p.join()

    # === display image ============================================================================

    # get maximum and minimum value of each data
    _vmaxs = [data.max() for data in list_data]
    _vmins = [data.min() for data in list_data]

    # define vmaxs
    if isinstance(vmax, (float, int)):
        vmaxs: list[float] = [vmax for _ in range(len(list_data))]
    else:
        vmaxs: list[float] = _vmaxs

    # define vmins
    if isinstance(vmin, (float, int)):
        vmins: list[float] = [vmin for _ in range(len(list_data))]
    else:
        vmins: list[float] = _vmins

    if cbar_mode == "single":
        vmaxs: list[float] = [max(vmaxs) for _ in range(len(vmaxs))]
        vmins: list[float] = [min(vmins) for _ in range(len(vmins))]

    # r, z grids
    r_pts = np.linspace(rmin, rmax, nr)
    z_pts = np.linspace(zmin, zmax, nz)

    for i in range(len(profiles)):
        norm = set_norm(plot_mode, vmins[i], vmaxs[i], linear_width=linear_width)

        # mapping
        mappable = grids[i].pcolormesh(
            r_pts, z_pts, profiles[i], cmap=cmap, shading="auto", norm=norm
        )

        # annotation of toroidal angle
        if isinstance(labels, Collection) and len(labels) >= len(profiles):
            add_inner_title(grids[i], f"{labels[i]}")

        # set each axis properties
        set_axis_properties(grids[i])

    # create colorbar objects and store them into a list
    if cbar_mode == "each":
        for i, grid in enumerate(grids.axes_all):
            extend = _set_cbar_extend(vmins[i], vmaxs[i], _vmins[i], _vmaxs[i])
            cbar = plt.colorbar(grid.images[0], grids.cbar_axes[i], extend=extend)
            _linear_width = (
                _set_leaner_width(vmins[i], vmaxs[i]) if linear_width is None else linear_width
            )
            set_axis_format(cbar.ax.yaxis, cbar_format, linear_width=_linear_width)
            cbar.set_label(clabels[i])

    else:  # cbar_mode == "single"
        vmax, vmin = max(vmaxs), min(vmins)
        _vmax, _vmin = max(_vmaxs), min(_vmins)
        extend = _set_cbar_extend(vmin, vmax, _vmin, _vmax)
        norm = set_norm(plot_mode, vmin, vmax, linear_width=linear_width)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable, grids.cbar_axes[0], extend=extend)
        _linear_width = _set_leaner_width(vmin, vmax) if linear_width is None else linear_width
        set_axis_format(cbar.ax.yaxis, cbar_format, linear_width=_linear_width)
        cbar.set_label(clabels[0])

    # set axis labels
    nrow, ncol = grids.get_geometry()
    for i in range(nrow):
        grids[i * ncol].set_ylabel("$Z$ [m]")
    for i in range(ncol):
        grids[i + (nrow - 1) * ncol].set_xlabel("$R$ [m]")

    return (fig, grids)


def show_basis_profiles(
    bases_array: np.ndarray,
    func1: Callable[[float, float, float], Real],
    func2: Callable[[float, float, float], Real],
    bins: int = 29700,
    basis_indices: list[int] = BASIS_INDICES,
    phi_degs: np.ndarray = PHIS5,
    mask: MaskMode | None = "grid",
    fig: Figure | None = None,
    cmap: str | Colormap = "RdBu_r",
    plot_mode: PlotMode = "asinh",
    linear_width: float | None = None,
    show_phi: bool = True,
    **kwargs,
) -> tuple[Figure, ImageGrid]:
    """Show several bases profiles in R-Z plane with several toroidal angles.

    Parameters
    ----------
    bases_array : np.ndarray
        Array of bases. The shape of the array must be (bins, n_bases).
    func1 : Callable[[float, float, float], Real]
        Callable object for the first zone :math:`\\phi \\in [0, 9]`.
    func2 : Callable[[float, float, float], Real]
        Callable object for the second zone :math:`\\phi \\in [9, 18]`.
    bins : int, optional
        Number of bins, by default 29700.
    basis_indices : list[int], optional
        List of basis indices, by default [0, 1, 2, 3, 4, 5].
    phi_degs : np.ndarray, optional
        Toroidal angles, by default `np.linspace(0, 17.99, 5)`.
    mask : {"wall", "grid", "<0", "<=0"}, optional
        Masking profile, by default "grid".
    fig : Figure, optional
        Figure object, by default None.
    cmap : str | Colormap, optional
        Colorbar map, by default "RdBu_r".
    plot_mode : {"scalar", "log", "centered", "symlog", "asinh"}, optional
        Which scale to adapt to the colormap, by default "asinh".
        Each mode corresponds to the `~matplotlib.colors.Normalize` object.
    linear_width : float, optional
        Linear width of ``asinh/symlog`` norm, by default None.
        If None, the linear width is automatically set to two digit below the absolute maximum or
        minimum value of the data.
    show_phi : bool, optional
        If True, toroidal angle is annotated in each axis, by default True.
    **kwargs : :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        User-specified properties, by default `axes_pad=0.0`, `label_mode="L"` and
        `cbar_mode=None`.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure object.
    grids : `~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`
        ImageGrid object.
    """
    # set grid object
    from ..emc3.grid import Grid

    grid1, grid2 = Grid("zone0"), Grid("zone11")

    # sampling rate
    resolution = 1.0e-3
    rmin, rmax = 2.5, 5.0
    zmin, zmax = -1.3, 1.3
    nr = round((rmax - rmin) / resolution)
    nz = round((zmax - zmin) / resolution)

    # figure object
    if not isinstance(fig, Figure):
        width = 2.6 * len(phi_degs)
        height = 1.2 * len(basis_indices) * width / len(phi_degs)
        fig = plt.figure(dpi=200, figsize=(width, height))

    # set default ImageGrid parameters
    grid_params = dict(**kwargs)
    grid_params.setdefault("axes_pad", 0.0)
    grid_params.setdefault("label_mode", "L")
    grid_params.setdefault("cbar_mode", None)

    grids = ImageGrid(fig, 111, (len(basis_indices), len(phi_degs)), **grid_params)

    # === parallelized sampling ====================================================================
    manager = Manager()
    profiles = manager.dict()
    job_queue = manager.Queue()

    # create tasks
    i = 0
    for basis_index in basis_indices:
        for phi_deg in phi_degs:
            job_queue.put((i, phi_deg, basis_index))
            i += 1

    # produce worker pool
    pool_size = min(len(phi_degs) * len(basis_indices), cpu_count())
    workers = [
        Process(
            target=_worker3,
            args=(
                bases_array,
                bins,
                func1,
                func2,
                mask,
                (rmin, rmax, nr),
                (zmin, zmax, nz),
                job_queue,
                profiles,
            ),
        )
        for _ in range(pool_size)
    ]
    for p in workers:
        p.start()

    for p in workers:
        p.join()

    # ==============================================================================================

    # maximum value of all profiles
    vmin = np.amin([profile.min() for profile in profiles.values()])
    vmax = np.amax([profile.max() for profile in profiles.values()])

    # set norm
    norm = set_norm(plot_mode, vmin, vmax, linear_width=linear_width)

    # r, z grids
    r_pts = np.linspace(rmin, rmax, nr)
    z_pts = np.linspace(zmin, zmax, nz)

    for i, ax in enumerate(grids.axes_all):
        # mapping
        ax.pcolormesh(r_pts, z_pts, profiles[i], cmap=cmap, shading="auto", norm=norm)

        # annotation of toroidal angle
        if show_phi:
            add_inner_title(ax, f"$\\phi=${phi_degs[i % len(phi_degs)]:.1f}°")

        # set each axis properties
        set_axis_properties(ax)

        # plot outline
        if (phi := phi_degs[i % len(phi_degs)]) < 9.0:
            grid1.plot_outline(phi, fig=fig, ax=ax, show_phi=False)
        else:
            grid2.plot_outline(phi, fig=fig, ax=ax, show_phi=False)

    # set axis labels
    nrow, ncol = grids.get_geometry()
    for i in range(nrow):
        grids[i * ncol].set_ylabel("$Z$ [m]")
    for i in range(ncol):
        grids[i + (nrow - 1) * ncol].set_xlabel("$R$ [m]")

    return (fig, grids)


def _worker1(
    func: Callable[[float, float, float], Real],
    mask: MaskMode | None,
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
    index_func: Callable[[float, float, float], int],
    mask: MaskMode | None,
    r_range: tuple[float, float, int],
    z_range: tuple[float, float, int],
    job_queue: Queue,
    profiles: dict,
) -> None:
    """Worker process to generate sampled & masked profiles."""
    from ..emc3.cython.mapper import Mapper

    while not job_queue.empty():
        try:
            # extract a task
            index, data = job_queue.get(block=False)

            # generate mapper function
            func = Mapper(index_func, data)

            # generate profile
            profile = _sampler(func, phi_deg, mask, r_range, z_range)

            profiles[index] = profile

        except Exception:
            break


def _worker3(
    bases_array: np.ndarray,
    bins: int,
    index_func1: Callable[[float, float, float], int],
    index_func2: Callable[[float, float, float], int],
    mask: MaskMode | None,
    r_range: tuple[float, float, int],
    z_range: tuple[float, float, int],
    job_queue: Queue,
    profiles: dict,
) -> None:
    """Worker process to generate sampled & masked profiles."""
    from ..emc3.cython.mapper import Mapper

    while not job_queue.empty():
        try:
            # extract a task
            index, phi_deg, basis_index = job_queue.get(block=False)

            # generate mapper function
            if phi_deg < 9.0:
                func = Mapper(index_func1, bases_array[:bins, basis_index])
            else:
                func = Mapper(index_func2, bases_array[bins:, basis_index])

            # generate profile
            profile = _sampler(func, phi_deg, mask, r_range, z_range)

            profiles[index] = profile

        except Exception:
            break


def _sampler(
    func: Callable[[float, float, float], Real],
    phi_deg: float,
    mask: MaskMode | None,
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
                mask_arr = np.zeros_like(sampled, dtype=bool)

        case "<0":
            mask_arr = sampled < 0

        case "<=0":
            mask_arr = sampled <= 0

        case _:
            mask_arr = np.zeros_like(sampled, dtype=bool)

    # generate masked sampled array
    profile: np.ndarray = np.transpose(np.ma.masked_array(sampled, mask=mask_arr))

    return profile


def show_profile_xy_plane(
    func: Callable[[float, float, float], Real],
    z: float = 0.0,
    fig: Figure | None = None,
    mask: MaskMode | None = "grid",
    vmax: float | None = None,
    vmin: float | None = 0.0,
    resolution: float = 1.0e-3,
    xy_range: tuple[float, float, float, float] = (2.5, 4.5, 0, 1.5),
    clabel: str = "",
    cmap: str | Colormap = CMAP_RED,
    plot_mode: PlotMode = "scalar",
    cbar_format: FormatMode | PlotMode | None = None,
    linear_width: float | None = None,
    **kwargs,
) -> ImageGrid:
    """Show EMC3-EIRENE discretized data function in X-Y plane.

    Parameters
    ----------
    func : Callable[[float, float, float], Real]
        Callable object.
    z : float, optional
        Z-axis value, by default 0.0.
    fig : `~matplotlib.figure.Figure`, optional
        Figure object, by default None.
    mask : {"wall", "grid", "<0", "<=0"}, optional
        Masking profile, by default "grid".
    vmax : float, optional
        Maximum value of colorbar limits, by default None.
        If None, maximum value is chosen of all sampled values.
    vmin : float, optional
        Minimum value of colorbar limits, by default 0.0.
        If None, minimum value is chosen of all sampled values.
    resolution : float, optional
        Sampling resolution, by default 1.0e-3.
    xy_range : tuple[float, float, float, float], optional
        Sampling range : :math:`(X_\\text{min}, X_\\text{max}, Y_\\text{min}, Y_\\text{max})`,
        by default ``(2.5, 4.5, 0, 1.5)``.
    clabel : str, optional
        Colorbar label, by default "".
    cmap : str | `~matplotlib.colors.Colormap`, optional
        Colorbar map, by default `CMAP_RED` (custom Red colormap extracted from "RdBu_r").
    plot_mode : {"scalar", "log", "centered", "symlog", "asinh"}, optional
        Which scale to adapt to the colormap, by default "scalar".
        Each mode corresponds to the `~matplotlib.colors.Normalize` object.
    cbar_format : {"scalar", "log", "symlog", "asinh", "percent", "eng"}, optional
        Formatter to colorbar's major y-axis locator, by default None.
        If None, the formatter is automatically set to the same one determined by `plot_mode`.
    linear_width : float, optional
        Linear width of ``asinh/symlog`` norm, by default None.
        If None, the linear width is automatically set to two digit below the absolute maximum or
        minimum value of the data.
    **kwargs : :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid` properties, optional
        User-specified properties, by default `axes_pad=0.0`, `label_mode="L"` and
        `cbar_mode="single"`.

    Returns
    -------
    :obj:`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`
        ImageGrid object.
    """
    # sampling rate
    xmin, xmax, ymin, ymax = xy_range
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid xy_range.")

    nx = round((xmax - xmin) / resolution)
    ny = round((ymax - ymin) / resolution)

    # sampling
    x_pts, y_pts, samples = sample_xy_plane(func, (xmin, xmax, nx), (ymin, ymax, ny), z=z)

    # masking samples
    match mask:
        case "<=0":
            samples = np.ma.masked_less_equal(samples, 0.0)
        case "<0":
            samples = np.ma.masked_less(samples, 0.0)
        case "grid":
            if inside_grids := getattr(func, "inside_grids", None):
                _, _, mask_arr = sample_xy_plane(
                    inside_grids, (xmin, xmax, nx), (ymin, ymax, ny), z=z
                )
                mask_arr = np.logical_not(mask_arr)
                samples = np.ma.masked_array(samples, mask=mask_arr)
            else:
                pass
        case _:
            pass

    # set vmin, vmax
    vmin = samples.min() if vmin is None else vmin
    vmax = samples.max() if vmax is None else vmax

    if not isinstance(fig, Figure):
        fig = plt.figure(dpi=150)

    grids = ImageGrid(
        fig, 111, nrows_ncols=(1, 1), axes_pad=0.0, label_mode="L", cbar_mode="single"
    )

    # set norm
    norm = set_norm(plot_mode, vmin, vmax, linear_width=linear_width)

    # plot
    mappable = grids[0].pcolormesh(x_pts, y_pts, samples.T, cmap=cmap, norm=norm)

    # set colorbar
    cbar = plt.colorbar(mappable, cax=grids.cbar_axes[0])

    # set colorbar's locator and formatter
    if cbar_format is None:
        cbar_format = plot_mode

    if linear_width is None:
        linear_width = _set_leaner_width(vmin, vmax)

    set_axis_format(cbar.ax.yaxis, cbar_format, linear_width=linear_width, **kwargs)
    cbar.set_label(clabel)

    # set axis properties
    grids[0].set_xlabel("$X$ [m]")
    grids[0].set_ylabel("$Y$ [m]")
    grids[0].xaxis.set_major_locator(MultipleLocator(0.5))
    grids[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    grids[0].yaxis.set_major_locator(MultipleLocator(0.5))
    grids[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    grids[0].tick_params(direction="in", labelsize=10, which="both", top=True, right=True)

    # plot toroidal angle lines
    grids[0].axline((0, 0), slope=np.tan(np.deg2rad(9.0)), color="k", linestyle="--")
    grids[0].axline((0, 0), slope=np.tan(np.deg2rad(18.0)), color="k", linestyle="--")

    # toroidal angle text
    grids[0].text(2.7, 0.45, "$\\phi=9°$", rotation=9, fontsize=10)
    grids[0].text(2.7, 0.9, "$\\phi=18°$", rotation=18, fontsize=10)

    # plot magnetic axis (r, z) = (3.6, 0.0)
    y_axis = np.linspace(ymin, ymax, 100)
    phis = np.arcsin(y_axis / 3.6)
    x_axis = 3.6 * np.cos(phis)
    grids[0].plot(x_axis, y_axis, color="k", linestyle="--")

    # axes limit
    grids[0].set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    return grids


def set_axis_properties(axes: Axes):
    """Set x-, y-axis property.

    This function set axis labels and tickers.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        Matplotlib Axes object.
    """
    axes.set_xlabel("$R$ [m]")
    axes.xaxis.set_minor_locator(MultipleLocator(0.1))
    axes.yaxis.set_minor_locator(MultipleLocator(0.1))
    axes.xaxis.set_major_formatter("{x:.1f}")
    axes.yaxis.set_major_formatter("{x:.1f}")
    axes.tick_params(direction="in", labelsize=10, which="both", top=True, right=True)


def set_norm(
    mode: PlotMode, vmin: float, vmax: float, linear_width: float | None = None
) -> Normalize:
    """Set variouse `~matplotlib.colors.Normalize` object.

    Parameters
    ----------
    mode : {"scalar", "log", "centered", "symlog", "asinh"}
        which scale to adapt to the colormap.
    vmin : float
        Minimum value of the profile.
    vmax : float
        Maximum value of the profile.
    linear_width : float, optional
        Linear width of asinh/symlog norm, by default None.
        If None, automatically set to two digit below the absolute value.

    Returns
    -------
    :obj:`~matplotlib.colors.Normalize`
        Normalize object corresponding to the mode.
    """
    # set norm
    absolute = max(abs(vmax), abs(vmin))
    if linear_width is None:
        linear_width = _set_leaner_width(vmin, vmax)

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


def set_axis_format(
    axis: XAxis | YAxis,
    formatter: FormatMode | PlotMode,
    linear_width: float = 1.0,
    offset_position: Literal["left", "right"] = "left",
    **kwargs,
) -> None:
    """Set axis format.

    Set specified axis major formatter and both corresponding major and minor locators.

    Parameters
    ----------
    axis : `~matplotlib.axis.XAxis` | `~matplotlib.axis.YAxis`
        Matplotlib axis object.
    formatter : {"scalar", "log", "symlog", "asinh", "percent", "eng"}
        Formatter mode of the axis. Values in non-implemented modes are set to
        `~matplotlib.ticker.ScalarFormatter` with ``useMathText=True``.
    linear_width : float, optional
        Linear width of asinh/symlog norm, by default 1.0.
    offset_position : {"left", "right"}, optional
        Position of the offset text like :math:`\\times 10^3`, by default ``"left"``.
        This parameter only affects `~matplotlib.axis.YAxis` object.
    **kwargs
        Keyword arguments for formatter.
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

    # set axis properties
    if isinstance(axis, YAxis):
        axis.set_offset_position(offset_position)
    axis.set_major_formatter(fmt)
    axis.set_major_locator(major_locator)
    axis.set_minor_locator(minor_locator)


def _set_cbar_extend(user_vmin: float, user_vmax: float, data_vmin: float, data_vmax: float) -> str:
    """Set colorbar's extend.

    Parameters
    ----------
    user_vmin : float
        User defined minimum value.
    user_vmax : float
        User defined maximum value.
    data_vmin : float
        Minimum value of the profile.
    data_vmax : float
        Maximum value of the profile.

    Returns
    -------
    str
        Colorbar's extend.
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


def _set_leaner_width(vmin: float, vmax: float) -> float:
    """Set linear width for asinh/symlog norm.

    Set linear width to two digit below the absolute value. (e.g. 0.001 if vmax=3.1 or vmin=-4.9)

    Parameters
    ----------
    vmin : float
        Minimum value of the profile.
    vmax : float
        Maximum value of the profile.

    Returns
    -------
    float
        Linear width of asinh/symlog norm.
    """
    return 10 ** (np.trunc(np.log10(max(abs(vmax), abs(vmin)))) - 2)


def add_inner_title(
    ax: Axes,
    title: str,
    loc: str = "upper left",
    size: float = plt.rcParams["legend.fontsize"],
    borderpad: float = 0.5,
    **kwargs,
):
    """Add inner title to the axes.

    The text is padded by borderpad and has white stroke effect.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes object.
    title : str
        Title text.
    loc : str, optional
        Location of the title, by default "upper left".
    size : int, optional
        Font size of the title, by default `plt.rcParams["legend.fontsize"]`.
    borderpad : float, optional
        Padding of the title, by default 0.5.
    **kwargs
        Keyword arguments for `~matplotlib.offsetbox.AnchoredText`.

    Returns
    -------
    `~matplotlib.offsetbox.AnchoredText`
        AnchoredText object.
    """
    prop = dict(path_effects=[withStroke(linewidth=3, foreground="w")], size=size)
    at = AnchoredText(
        title, loc=loc, prop=prop, pad=0.0, borderpad=borderpad, frameon=False, **kwargs
    )
    ax.add_artist(at)
    return at
