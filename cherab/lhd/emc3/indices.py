"""This module provides functions to create index function for EMC3."""

from typing import Literal, get_args

import h5py  # noqa: F401
import numpy as np
import xarray as xr
from raysect.core.math import triangulate2d

# from raysect.primitive.mesh import TetraMeshData
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..tools.fetch import fetch_file
from .cython import Discrete3DMesh
from .cython.tetra_mesh import TetraMeshData
from .grid import Grid

__all__ = ["load_index_func", "triangulate"]

INDEX_TYPES = Literal["cell", "physics", "coarse"]

ZONE_MATCH = {
    "zone1": "zone2",
    "zone2": "zone1",
    "zone3": "zone4",
    "zone4": "zone3",
    "zone12": "zone15",
    "zone15": "zone12",
    "zone13": "zone14",
    "zone14": "zone13",
}


def load_index_func(
    zones: list[str],
    index_type: INDEX_TYPES = "cell",
    load_tetra_mesh: bool = True,
    dataset: str = "emc3/grid-360.nc",
    quiet: bool = False,
    **kwargs,
) -> tuple[Discrete3DMesh | dict[str, np.ndarray], dict[str, int]]:
    """Load index function of EMC3-EIRENE mesh.

    Parameters
    ----------
    zones : list[str]
        List of zone names. The order of the zones is important.
        All zone names must be unique.
    index_type : {"cell", "physics", "coarse"}, optional
        Index type, by default ``"cell"``.
    load_tetra_mesh : bool, optional
        Whether to load a pre-created tetrahedral mesh, by default is True.
    dataset : str, optional
        Dataset name, by default ``"emc3/grid-360.nc"``.
    quiet : bool, optional
        Mute status messages, by default False.
    **kwargs
        Keyword arguments to pass to `.fetch_file`.

    Returns
    -------
    tuple[`.Discrete3DMesh` | dict[str, ndarray], dict[str, int]]
        If ``load_tetra_mesh==True``, returns a index function and dictionary of voxel numbers for
        each zone.
        If ``load_tetra_mesh==False``, returns a dictionary of index arrays and dictionary of
        voxel numbers for each zone.

    Examples
    --------
    >>> index_func, bins = load_index_func(["zone0", "zone11"], index_type="coarse")
    >>> index_func
    IntegerFunction3D()
    >>> bins
    {'zone0': 29700, 'zone11': 29700}

    In case of ``load_tetra_mesh=False``:

    >>> index_arrays, bins = load_index_func(["zone0"], index_type="coarse", load_tetra_mesh=False)
    >>> index_arrays
    {'zone0': array([[[    0,     0,     0, ..., 26400, 26400, 26400],
                      [    0,     0,     0, ..., 26400, 26400, 26400],
                      [    0,     0,     0, ..., 26400, 26400, 26400],
                      [ 3299,  3299,  3299, ..., 29699, 29699, 29699]]], dtype=uint32)}
    """
    # Validate parameters
    if len(zones) != len(set(zones)):
        raise ValueError(
            "Duplicate elements found in the zones list. All zone names must be unique."
        )
    if index_type not in set(get_args(INDEX_TYPES)):
        raise ValueError(f"Invalid index_type: {index_type}")

    # Initialize progress bar
    console = Console(quiet=quiet)
    progress = Progress(
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn("simpleDots"),
        console=console,
    )
    task_id = progress.add_task("", total=1)

    with progress:
        # ==========================
        # === Fetch grid dataset ===
        # ==========================
        progress.update(task_id, description="Fetching grid dataset")
        path = fetch_file(dataset, **kwargs)
        groups = xr.open_groups(path)

        # =========================
        # === Calculate indices ===
        # =========================
        progress.update(task_id, description="Retrieving indices")
        dict_index1: dict[str, np.ndarray] = {}
        dict_num_voxels: dict[str, int] = {}
        start_index = 0
        for zone in zones:
            data = groups[f"/{zone}/index"][index_type].data
            _max_index = data.max() + 1
            dict_num_voxels[zone] = _max_index

            if index_type == "physics":
                dict_index1[zone] = data
            else:
                dict_index1[zone] = data + start_index
                start_index += _max_index

        if not load_tetra_mesh:
            progress.update(task_id, visible=False, advance=1, refresh=True)
            return dict_index1, dict_num_voxels

        # =======================
        # === Load tetra mesh ===
        # =======================

        # Combine file names from zones
        # NOTE: Allowing to order zone's name.
        # That is, "zone0+zone11" and "zone11+zone0" are considered different
        # and both combinations are permitted.
        zones_str = "+".join(zones)
        progress.update(task_id, description=f"Loading {zones_str}.rsm")
        path_tetra = fetch_file(f"tetra/{zones_str}.rsm", **kwargs)
        tetra_mesh = TetraMeshData.from_file(path_tetra)

        # ====================
        # === Sort indices ===
        # ====================
        # Create second index array when toroidal angle is out of range [0, 18] in degree
        progress.update(task_id, description="Sorting indices")
        match index_type:
            case "physics":
                dict_index2 = dict_index1

            case "cell" | "coarse":
                dict_index2: dict[str, np.ndarray] = {}
                for zone in zones:
                    if zone in {"zone0", "zone11"}:
                        dict_index2[zone] = dict_index1[zone][:, ::-1, :]
                    else:
                        # TODO: Needs to be scrutinized.
                        if (value := dict_index1.get(ZONE_MATCH[zone], None)) is not None:
                            dict_index2[zone] = value
                        else:
                            dict_index2[zone] = groups[f"/{zone}/index"][index_type].data
            case _:
                raise NotImplementedError(f"'{index_type}' is not implemented yet.")

        # Vectorize and Concatenate
        index1_1d = np.hstack([dict_index1[zone].ravel(order="F") for zone in zones])
        index2_1d = np.hstack([dict_index2[zone].ravel(order="F") for zone in zones])

        # Finalize progress bar
        progress.update(
            task_id,
            description=f"[bold green]Loaded index function[/bold green] ({'+'.join(zones)})",
            advance=1,
            refresh=True,
        )

    # TODO: Needs to consider indices3/indices4 for edge zones?
    discrete3d = Discrete3DMesh(tetra_mesh, index1_1d, index2_1d)

    return discrete3d, dict_num_voxels


def triangulate(
    grid: Grid,
    phi: float | None = None,
    n_phi: int = 0,
    index_type: INDEX_TYPES = "coarse",
):
    """Triangulate grid data at a specific toroidal angle.

    This method returns vertices and triangles at ploidal plane (R-Z plane) at a specific
    toroidal angle.
    The grid is interpolated linearly between two nearest toroidal angles if specifying
    arbitrary toroidal angle `phi`.
    Also, indices of triangles are returned according to the index type.

    Parameters
    ----------
    grid : `.Grid`
        Grid object.
    phi : float, optional
        Toroidal angle in [degree], by default None.
        If specified, the grid is interpolated linearly and prioritized over `n_phi`.
    n_phi : int, optional
        Index of toroidal angle, by default 0.
        If specified, the grid is not interpolated and used as is.
    index_type : {"cell", "physics", "coarse"}, optional
        Index type, by default "coarse".

    Returns
    -------
    verts : (N, 2) ndarray
        Vertices at poloidal plane. N is the number of vertices.
    tri : (M, 3) ndarray
        Triangles at poloidal plane. M is the number of triangles.
    indices : (M,) ndarray
        Indices of triangles.

    Examples
    --------
    >>> grid = Grid("zone0")
    >>> verts, tris, indices = triangulate(grid, phi=4.2)
    >>> verts.shape
    (55200, 2)
    >>> tri.shape
    (48600, 3)
    >>> indices.shape
    (48600,)
    """
    if isinstance(phi, (int, float)):
        angles = grid.data_array["ζ"].data
        if phi < angles[0] or phi > angles[-1]:
            raise ValueError(f"Toroidal angle {phi} is out of range. [{angles[0]}, {angles[-1]}]")
    elif isinstance(n_phi, int):
        angles = grid.data_array["ζ"].data
        if n_phi < 0 or n_phi >= len(angles):
            raise IndexError(f"Index of toroidal angle {n_phi} is out of range. [0, {len(angles)}]")
    else:
        raise ValueError("Either `phi` or `n_phi` must be specified.")

    # --------------------------
    # === Load index dataset ===
    # --------------------------
    groups = xr.open_groups(grid.path)
    match index_type:
        case "coarse":
            ds_index = groups[f"/{grid.zone}/index"]["coarse"]
            indices_radial = ds_index.attrs["indices_radial"]
            indices_poloidal = ds_index.attrs["indices_poloidal"]

        case "cell" | "physics":
            L, M, _ = grid.shape
            ds_index = groups[f"/{grid.zone}/index"][index_type]
            indices_radial = [i for i in range(0, L)]
            indices_poloidal = [i for i in range(0, M)]

        case _:
            raise ValueError(f"Invalid index type: {index_type}")

    # ------------------------
    # === Interpolate grid ===
    # ------------------------
    if isinstance(phi, (int, float)):
        grid_rz = grid.data_array.interp(ζ=phi, method="linear")
        index_rz = ds_index.sel(ζ=phi, method="nearest")

    else:
        grid_rz = grid.data_array.isel(ζ=n_phi)
        index_rz = ds_index.isel(ζ=n_phi)

    # ------------------------
    # === Triangulate grid ===
    # ------------------------
    list_verts = []
    list_tris = []
    list_indices = []

    start_triangle = 0
    for l, m in np.ndindex(len(indices_radial) - 1, len(indices_poloidal) - 1):  # noqa: E741
        m0, m1 = indices_poloidal[m], indices_poloidal[m + 1]
        l0, l1 = indices_radial[l], indices_radial[l + 1]

        # Polygon's vertices
        if l0 == 0:  # Remove coincident points
            _verts = np.vstack(
                (
                    grid_rz[l0:l1, m0],
                    grid_rz[l1, m0:m1],
                    grid_rz[l1:l0:-1, m1],
                )
            )
        else:
            _verts = np.vstack(
                (
                    grid_rz[l0:l1, m0],
                    grid_rz[l1, m0:m1],
                    grid_rz[l1:l0:-1, m1],
                    grid_rz[l0, m1:m0:-1],
                )
            )

        # Triangulate polygon
        _tris = triangulate2d(_verts)

        # Retrieve indices
        index = index_rz.isel(ρ=l0, θ=m0).item()

        # Store values temporarily
        list_verts.append(_verts)
        list_tris.append(_tris + start_triangle)
        list_indices.append(np.full(_tris.shape[0], index, dtype=np.int32))

        start_triangle += _verts.shape[0]

    # Return concatenated values
    return np.vstack(list_verts), np.vstack(list_tris), np.hstack(list_indices)
