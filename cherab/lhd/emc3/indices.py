"""This module provides functions to create index function for EMC3."""
import h5py
import numpy as np
from numpy.typing import NDArray
from raysect.core.math import triangulate2d
from raysect.core.math.function.float import Discrete2DMesh
from raysect.primitive.mesh import TetraMesh

from .cython import Discrete3DMesh
from .grid import Grid
from .repository.utility import DEFAULT_HDF5_PATH, DEFAULT_TETRA_MESH_PATH

__all__ = ["create_index_func", "create_new_index", "create_2d_mesh"]


def create_index_func(
    zone: str, index_type="cell", load_tetra_mesh: bool = True
) -> tuple[Discrete3DMesh, int] | int:
    """Create index function using :obj:`.Discrete3DMesh`

    The index data must be stored in ``grid-360`` group in HDF5 file.

    Parameters
    ----------
    zone
        zone name
    index_type
        index type, by default ``"cell"``.
        usable types are ``"cell"``, ``"physics"`` and ``"coarse"``.
    load_tetra_mesh
        whether to load tetra mesh, by default True.
        If False, return only number of indices (bins).

    Returns
    -------
    tuple[Discrete3DMesh, int] | int
        index function and number of indices (bins) or only number of indices (bins)
    """
    _create_index = False
    with h5py.File(DEFAULT_HDF5_PATH, mode="r+") as file:
        try:
            indices: NDArray[np.uint32] = file["grid-360"][zone]["index"][index_type][:]

        except Exception:
            _create_index = True

    if _create_index:
        indices = create_new_index(index_type, zone)

    # load tetra mesh
    if load_tetra_mesh:
        if (rsm_path := DEFAULT_TETRA_MESH_PATH / f"{zone}.rsm").exists():
            tetra = TetraMesh.from_file(rsm_path)
        else:
            raise FileNotFoundError(f"{rsm_path.name} file does not exist.")
    else:
        return indices.max() + 1

    # create indices when phi is out of range [0, 18] in degree
    index2 = indices[:, ::-1, :]

    # vectorize index array
    index1_1d = indices.ravel(order="F")
    index2_1d = index2.ravel(order="F")

    return Discrete3DMesh(tetra, index1_1d, index2_1d), indices.max() + 1


def create_new_index(index_type: str, zone: str) -> NDArray[np.uint32]:
    """Create new index array.

    The created index data will be stored in HDF5 file at :obj:`.DEFAULT_HDF5_PATH`.

    Parameters
    ----------
    index_type
        index type. {`"coars"`} is only supported now.
    zone
        zone name

    Returns
    -------
    NDArray[np.uint32]
        index array
    """

    grid = Grid(zone)
    L, M, N = grid.shape

    match index_type:
        case "coarse":
            # reduce the resolution of the grid
            # TODO: consider others except for zone0 and zone11
            radial_slice = [17, 41, L]
            radial_step = [1, 4, 4]
            poloidal_step = 6
            toroidal_step = 4
            radial_indices = (
                [i for i in range(0, radial_slice[0], radial_step[0])]
                + [i for i in range(radial_slice[0], radial_slice[1], radial_step[1])]
                + [i for i in range(radial_slice[1], radial_slice[2], radial_step[2])]
            )
            poloidal_indices = [i for i in range(0, M, poloidal_step)]
            toroidal_indices = [i for i in range(0, N, toroidal_step)]

            num_radial_index = len(radial_indices) - 1
            num_poloidal_index = len(poloidal_indices) - 1
            num_toroidal_index = len(toroidal_indices) - 1

            indices = np.zeros((L - 1, M - 1, N - 1), dtype=np.uint32)

            for i in range(num_radial_index):
                for j in range(num_poloidal_index):
                    for k in range(num_toroidal_index):
                        indices[
                            radial_indices[i] : radial_indices[i + 1],
                            poloidal_indices[j] : poloidal_indices[j + 1],
                            toroidal_indices[k] : toroidal_indices[k + 1],
                        ] = (
                            i + j * num_radial_index + k * num_radial_index * num_poloidal_index
                        )

            # save index array as dataset
            with h5py.File(DEFAULT_HDF5_PATH, mode="r+") as file:
                index_group = file["grid-360"][zone]["index"]
                ds = index_group.create_dataset(name=index_type, data=indices)

                # save attribution information
                ds.attrs["description"] = "coarse grid index"
                ds.attrs["shape description"] = "radial index, poloidal index, toroidal index"
                ds.attrs["L"] = L - 1
                ds.attrs["M"] = M - 1
                ds.attrs["N"] = N - 1
                ds.attrs["radial indices"] = radial_indices
                ds.attrs["poloidal indices"] = poloidal_indices
                ds.attrs["toroidal indices"] = toroidal_indices

        case _:
            raise ValueError(f"Invalid index type: {index_type}")

    return indices


def create_2d_mesh(
    zone: str, n_phi: int, index_type="coarse", default_value: int = -1
) -> tuple[Discrete2DMesh, int]:
    """:obj:`~raysect.core.math.function.float.function2d.interpolate.Discrete2DMesh` instance
    of the EMC3 poloidal plane.


    The index data must be stored in ``grid-360`` group in HDF5 file.

    Parameters
    ----------
    zone
        zone name
    n_phi
        index number of toroidal direction
    index_type
        index type, by default ``cell``
    default_value
        default mesh value, by default ``-1``

    Returns
    -------
    tuple[Discrete3DMesh, int]
        index function and number of indices
    """
    grid = Grid(zone)

    with h5py.File(DEFAULT_HDF5_PATH, mode="r+") as file:
        try:
            ds = file["grid-360"][zone]["index"][index_type]

            match index_type:
                case "coarse":
                    radial_indices: np.ndarray = ds.attrs["radial indices"]
                    poloidal_indices: np.ndarray = ds.attrs["poloidal indices"]

                case _:
                    L, M, _ = grid.shape
                    radial_indices: list = [i for i in range(0, L)]
                    poloidal_indices: list = [i for i in range(0, M)]

        except Exception as err:
            raise ValueError(f"Invalid index type: {index_type}") from err

    index = 0
    for m in range(0, len(poloidal_indices) - 1):
        for l in range(0, len(radial_indices) - 1):  # noqa: E741
            m0, m1 = poloidal_indices[m], poloidal_indices[m + 1]
            l0, l1 = radial_indices[l], radial_indices[l + 1]

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
    res = create_index_func("zone0", "cell")
