"""This module provides functions to create index function for EMC3."""

import h5py  # noqa: F401
import numpy as np
import xarray as xr
from raysect.core.math import triangulate2d
from raysect.core.math.function.float import Discrete2DMesh
from raysect.primitive.mesh import TetraMeshData

from ..tools.fetch import fetch_file
from .cython import Discrete3DMesh
from .grid import Grid

__all__ = ["load_index_func", "create_2d_mesh"]


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
    zone: str,
    index_type: str = "cell",
    load_tetra_mesh: bool = True,
    dataset: str = "emc3/grid-360.nc",
    **kwargs,
) -> tuple[Discrete3DMesh, int] | int:
    """Create index function using `.Discrete3DMesh`.

    Parameters
    ----------
    zone : {"zone0",..., "zone4", "zone11",..., "zone15"}
        Zone name.
    index_type : {"cell", "physics", "coarse"}, optional
        Index type, by default ``"cell"``.
    load_tetra_mesh : bool, optional
        Whether to load tetra mesh, by default True. If False, return only number of indices (bins).
    dataset : str, optional
        Dataset name, by default ``"emc3/grid-360.nc"``.
    **kwargs
        Keyword arguments to pass to `.fetch_file`.

    Returns
    -------
    tuple[Discrete3DMesh, int] | int
        Index function and number of indices (bins) or only number of indices (bins).
    """
    # Fetch index data
    path = fetch_file(dataset, **kwargs)
    groups = xr.open_groups(path)

    # Load tetrahedra mesh
    if load_tetra_mesh:
        path_tetra = fetch_file(f"tetra/{zone}.rsm", **kwargs)
        tetra = TetraMeshData.from_file(path_tetra)
    else:
        return groups[f"/{zone}/index"][index_type].data.max() + 1

    # Procedure for the specific index type
    indices = groups[f"/{zone}/index"][index_type].data
    if index_type == "physics":
        indices2 = indices
    else:
        # Create indices when phi is out of range [0, 18] in degree
        if zone in {"zone0", "zone11"}:
            indices2 = indices[:, ::-1, :]
        else:
            # Match corresponding zone's indices
            zone2 = ZONE_MATCH[zone]
            indices2 = groups[f"/{zone2}/index"][index_type].data

    # Vectorize index array
    index1_1d = indices.ravel(order="F")
    index2_1d = indices2.ravel(order="F")

    return Discrete3DMesh(tetra, index1_1d, index2_1d), indices.max() + 1


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
            indices_radial = groups[f"/{zone}/index/coarse"].attrs["indices_radial"]
            indices_poloidal = groups[f"/{zone}/index/coarse"].attrs["indices_poloidal"]

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


if __name__ == "__main__":
    res = load_index_func("zone0", "cell")
