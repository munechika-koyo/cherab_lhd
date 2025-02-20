"""This module provides functions to create index function for EMC3."""

from typing import Literal, get_args

import h5py  # noqa: F401
import numpy as np
import xarray as xr
from raysect.core.math import triangulate2d
from raysect.core.math.function.float import Discrete2DMesh
from raysect.primitive.mesh import TetraMeshData
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..tools.fetch import fetch_file
from .cython import Discrete3DMesh
from .grid import Grid

__all__ = ["load_index_func", "create_2d_mesh"]

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
            dict_index1[zone] = data + start_index
            start_index += _max_index

        if not load_tetra_mesh:
            progress.update(task_id, visible=False, advance=1)
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
        progress.update(task_id, description="[bold green]Loaded index function", advance=1)
        progress.refresh()

    # TODO: Needs to consider indices3/indices4 for edge zones?
    discrete3d = Discrete3DMesh(tetra_mesh, index1_1d, index2_1d)

    return discrete3d, dict_num_voxels


def create_2d_mesh(
    zone: str,
    n_phi: int,
    index_type="coarse",
    dataset: str = "emc3/grid-360.nc",
    default_value: int = -1,
) -> tuple[Discrete2DMesh, int]:
    """`~raysect.core.math.function.float.function2d.interpolate.Discrete2DMesh` instance of the
    EMC3 poloidal plane.

    Parameters
    ----------
    zone : str
        Zone name.
    n_phi : int
        Index number of toroidal direction.
    index_type : {"cell", "physics", "coarse"}, optional
        Index type, by default ``coarse``.
    dataset : str, optional
        Dataset name, by default ``"emc3/grid-360.nc"``.
    default_value : int, optional
        Default mesh value, by default ``-1``.

    Returns
    -------
    tuple[Discrete2DMesh, int]
        Index function and number of indices.
    """
    grid = Grid(zone, dataset=dataset)
    groups = xr.open_groups(fetch_file(dataset))

    match index_type:
        case "coarse":
            indices_radial = groups[f"/{zone}/index"]["coarse"].attrs["indices_radial"]
            indices_poloidal = groups[f"/{zone}/index"]["coarse"].attrs["indices_poloidal"]

        case "cell" | "physics":
            L, M, _ = grid.shape
            indices_radial = [i for i in range(0, L)]
            indices_poloidal = [i for i in range(0, M)]

        case _:
            raise ValueError(f"Invalid index type: {index_type}")

    index = 0
    for l, m in np.ndindex(len(indices_radial) - 1, len(indices_poloidal) - 1):  # noqa: E741
        m0, m1 = indices_poloidal[m], indices_poloidal[m + 1]
        l0, l1 = indices_radial[l], indices_radial[l + 1]

        # create polygon's vertices
        # dealing with l0 = 0 (remove coincident points)
        if l0 == 0:
            _verts = np.vstack(
                (
                    grid[l0:l1, m0, n_phi, 0:2],
                    grid[l1, m0:m1, n_phi, 0:2],
                    grid[l1:l0:-1, m1, n_phi, 0:2],
                )
            )
        else:
            _verts = np.vstack(
                (
                    grid[l0:l1, m0, n_phi, 0:2],
                    grid[l1, m0:m1, n_phi, 0:2],
                    grid[l1:l0:-1, m1, n_phi, 0:2],
                    grid[l0, m1:m0:-1, n_phi, 0:2],
                )
            )

        # triangulate polygon
        _triangles = triangulate2d(_verts)

        # set index value to all triangles
        _data = np.full(_triangles.shape[0], index, dtype=np.int32)

        if index == 0:
            verts = _verts.copy()
            triangles = _triangles.copy()
            data = _data.copy()
        else:
            verts = np.vstack((verts, _verts))
            triangles = np.vstack((triangles, _triangles + triangles.max() + 1))
            data = np.hstack((data, _data))

        index += 1

    # create 2d mesh
    mesh = Discrete2DMesh(verts, triangles, data, limit=False, default_value=default_value)

    return mesh, index
