"""
Module to offer helper functions to generate interpolate 3D function
"""
from pathlib import Path

import h5py
import numpy as np

from .repository.utility import DEFAULT_HDF5_PATH, DEFAULT_TETRA_MESH_PATH

cimport cython
from cython.parallel cimport prange
from libc.limits cimport INT_MIN
from numpy cimport ndarray, uint32_t
from raysect.primitive.mesh cimport TetraMesh

from .cython.discrete3dmesh cimport Discrete3DMesh
from .cython.intfunction cimport IntegerFunction3D

__all__ = ["_IndexBase", "CellIndex", "PhysIndex", "TomographyZone"]

cdef:
    list ZONES

ZONES = [
    "zone0", "zone1", "zone2", "zone3", "zone4",
    "zone11", "zone12", "zone13", "zone14", "zone15"
]


cdef class _IndexBase(IntegerFunction3D):
    """Base class for EMC3-EIRENE cell index function.

    This class populates an IntegerFunction3D instance which returns a cell index
    when :math:`(X, Y, Z)` arguments are given.
    In addition, this offers common methods for specific cell index class.
    :obj:`.create_indices` method should be overrided in subclasses.

    Parameters
    ----------
    zones : list[str]
        list of zone names, by default ``["zone0", ..., "zone4", "zone11", ... "zone15"]``
    grid_group : str
        grid group name stored in HDF5 file, by default ``"grid-360"``
    tetra_path : str
        path to the directory containing TetraMesh .rsm files, by default ``~/.cherab/lhd/tetra/``
    hdf5_path : str
        path to the data repository formatted as HDF5, by default ``~/.cherab/lhd/emc3.hdf5``
    populate : bool
        whether or not to populate instances of Discrete3DMesh, by default True
    """
    cdef:
        list _zones
        object _tetra_path
        object _hdf5_path
        str _grid_group
        list _interpolaters

    def __init__(
        self,
        list zones=ZONES,
        object tetra_path=DEFAULT_TETRA_MESH_PATH,
        object hdf5_path=DEFAULT_HDF5_PATH,
        str grid_group="grid-360",
        bint populate=True,
    ):
        # store parameters as attributes
        self._zones = zones
        self._interpolaters = []

        # set path config with validation
        self.tetra_path = tetra_path
        self.hdf5_path = hdf5_path
        self.grid_group = grid_group

        # construct interpolaters
        if populate:
            self.create_interpolate()

    @property
    def zones(self):
        """list[str]: list of zones
        """
        return self._zones

    @property
    def tetra_path(self):
        """:obj:`~pathlib.Path`: path to the directory containing TetraMesh .rsm files
        """
        return self._tetra_path

    @tetra_path.setter
    def tetra_path(self, path):
        if not isinstance(path, (Path, str)):
            raise TypeError("tetra_path must be a string or instance of pathlib.Path class.")
        if (tetra_path := Path(path)).exists():
            self._tetra_path = tetra_path
        else:
            raise FileExistsError(f"{tetra_path} does not exist.")


    @property
    def hdf5_path(self):
        """:obj:`~pathlib.Path`: path to the data repository formated as HDF5
        """
        return self._hdf5_path

    @hdf5_path.setter
    def hdf5_path(self, path):
        if not isinstance(path, (Path, str)):
            raise TypeError("hdf5_path must be a string or instance of pathlib.Path class.")
        if (hdf5_path := Path(path)).exists():
            self._hdf5_path = Path(path)
        else:
            raise FileExistsError(f"{hdf5_path} does not exist.")

    @property
    def grid_group(self):
        """str: grid group name stored in HDF5 file
        """
        return self._grid_group

    @grid_group.setter
    def grid_group(self, value):
        if not isinstance(value, str):
            raise TypeError("grid_group must be a string.")
        self._grid_group = value

    cpdef tuple create_indices(self):
        """Base method for creating indices array.

        Here all 4 indices are created and returned.

        Returns
        -------
        tuple[dict, dict, dict, dict]
            containing indices dictionaries.
        """
        cdef:
            dict indices1_dict = {}
            dict indices2_dict = {}
            dict indices3_dict = {}
            dict indices4_dict = {}

        return (indices1_dict, indices2_dict, indices3_dict, indices4_dict)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void create_interpolate(self):
        """Create indices array and populate instances of :obj:`~raysect.primitive.mesh.TetraMesh`.

        These instances are stored in ``self._interpolaters``
        """
        # create 1D indices arrays
        indices1_dict, indices2_dict, indices3_dict, indices4_dict = self.create_indices()

        for zone in self._zones:
            # load TetraMesh
            if (rsm_path := self._tetra_path / f"{zone}.rsm").exists():
                tetra = TetraMesh.from_file(rsm_path)

                # create interplater with Discrete3DMesh
                self._interpolaters.append(
                    Discrete3DMesh(
                        tetra,
                        indices1_dict[zone],
                        indices2_dict[zone],
                        indices3_dict[zone],
                        indices4_dict[zone],
                    )
                )
            else:
                raise FileExistsError(f"{rsm_path.name} does not exist in {self._tetra_path}.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:
        cdef:
            Discrete3DMesh func
            int value

        for func in self._interpolaters:
            value = func.evaluate(x, y, z)
            if value > -1:
                return value
        return -1


cdef class CellIndex(_IndexBase):
    """EMC3-EIRENE cell index function defined in zone0-4 & zone11-15.

    This class is a subclass of :obj:`~cherab.lhd.emc3.geometry._IndexBase` and
    populates callable instance returning a corresponding cell index
    when :math:`(X, Y, Z)` arguments are given.

    Parameters
    ----------
    **kwargs : :obj:`~cherab.lhd.emc3.geometry._IndexBase` properties, optional
        *kwargs* are used to specify properties like a `tetra_path`
    """
    def __init__(self, *args, **keywards):
        super().__init__(*args, **keywards)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple create_indices(self):
        cdef:
            dict cell_indices = {}
            int start = 0
            int last
            str zone
        with h5py.File(self.hdf5_path, mode="r") as h5file:
            for zone in self._zones:
                last = h5file[self._grid_group][zone]["grids"].attrs["num_cells"] + start
                cell_indices[zone] = np.arange(start=start, stop=last, dtype=np.uint32)
                start = last

        return (cell_indices, cell_indices, cell_indices, cell_indices)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef dict _create_mapping_array(self, dict cell_indices, dict grid_config):
        cdef:
            dict mapping_array = {}
            str zone
            int L, M, N

        for zone in self._zones:
            with h5py.File(self._hdf5_path, mode="r") as h5file:
                L = h5file[self._grid_group][zone]["grids"].attrs["L"]
                M = h5file[self._grid_group][zone]["grids"].attrs["M"]
                N = h5file[self._grid_group][zone]["grids"].attrs["N"]

            if zone in {"zone0", "zone11"}:
                mapping_array[zone] = self._cell_index2mapping_array(cell_indices[zone], L, M, N)
            elif zone in {"zone1", "zone3", "zone13"}:
                zone_next = "zone" + str(int(zone.split("zone")[1]) + 1)
                mapping_array[zone] = self._cell_index2mapping_array(
                    cell_indices[zone_next], L, M, N
                )
            elif zone in {"zone2", "zone4", "zone14"}:
                zone_prev = "zone" + str(int(zone.split("zone")[1]) - 1)
                mapping_array[zone] = self._cell_index2mapping_array(
                    cell_indices[zone_prev], L, M, N
                )
            elif zone == "zone12":
                zone_next = "zone15"
                mapping_array[zone] = self._cell_index2mapping_array(
                    cell_indices[zone_next], L, M, N
                )
            elif zone == "zone15":
                zone_prev = "zone12"
                mapping_array[zone] = self._cell_index2mapping_array(
                    cell_indices[zone_prev], L, M, N
                )

        return mapping_array

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef ndarray[uint32_t, ndim=1] _cell_index2mapping_array(
        self,
        uint32_t[::1] cell_indices,
        int L,
        int M,
        int N,
    ):return np.ravel(
            np.flip(
                np.reshape(cell_indices, (N - 1, M - 1, L - 1)),
                axis=1,
            )
        )


cdef class PhysIndex(_IndexBase):
    """EMC3-EIRENE-defined Physical Cell Index function.

    This class is a subclass of :obj:`~cherab.lhd.emc3.geometry._IndexBase` and
    populates callable instance returning a corresponding physical cell index
    when :math:`(X, Y, Z)` arguments are given.
    Physical cell indices must be stored in HDF5 dataset file which is specified by `hdf5_path`.

    Parameters
    ----------
    **kwargs : :obj:`~cherab.lhd.emc3.geometry._IndexBase` properties, optional
        *kwargs* are used to specify properties like a `tetra_path`
    """
    def __init__(self, **keywards):
        super().__init__(**keywards)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple create_indices(self):
        """All 4 indices are created and returned.

        Each index array is loaded from the HDF5 file at `hdf5_path`.

        Returns
        -------
        tuple[dict, dict, dict, dict]
            tuple containing indices dictionaries.
        """
        cdef:
            dict indices_dict = {}
            str zone

        with h5py.File(self.hdf5_path, mode="r") as h5file:
            for zone in self._zones:
                indices_dict[zone] = h5file[self._grid_group][zone]["index"][:]

        return (indices_dict, indices_dict, indices_dict, indices_dict)


cdef class TomographyZone(_IndexBase):
    """EMC3-EIRENE-based Tmography Zone function.

    This class is a subclass of :obj:`~cherab.lhd.emc3.geometry._IndexBase` and
    populates callable instance returning a corresponding tomography zone index
    when :math:`(X, Y, Z)` arguments are given.
    Total zone size is 252 = 14 (poloidal) x 18 (0-18 deg in toroidal).

    Parameters
    ----------
    **kwargs : :obj:`~cherab.lhd.emc3.geometry._IndexBase` properties, optional
        *kwargs* are used to specify properties like a `tetra_path`
    """
    def __init__(self, **keywards):
        super().__init__(**keywards)

    cpdef tuple create_indices(self):
        """All 4 indices are created and returned.

        Each index is created in each private method.

        Returns
        -------
        tuple[dict, dict, dict, dict]
            containing indices dictionaries.
        """
        cdef:
            dict indices1_dict = self._indices1()
            dict indices2_dict = self._indices2()
            dict indices3_dict = self._indices3()
            dict indices4_dict = self._indices4()

        return (indices1_dict, indices2_dict, indices3_dict, indices4_dict)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef dict _indices1(self):
        cdef:
            dict indices_dict = {}
            str zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [10, 11, 12, 13]  # zone1, 2, 3, 4
            int[4] indices_legs2 = [10, 11, 12, 13]  # zone12, 13, 14, 15
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self._zones:
            with h5py.File(self._hdf5_path, mode="r") as h5file:
                ds_grids = h5file[self._grid_group][zone]["grids"]
                L = ds_grids.attrs["L"]
                M = ds_grids.attrs["M"]
                N = ds_grids.attrs["N"]
                NUM_CELLS = ds_grids.attrs["num_cells"]

            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 48:
                                    indices_mv[i] = 4 + val
                                elif 48 <= m < 170:
                                    indices_mv[i] = 5 + val
                                elif 170 <= m < 300:
                                    indices_mv[i] = 6 + val
                                elif 300 <= m < 430:
                                    indices_mv[i] = 7 + val
                                elif 430 <= m < 552:
                                    indices_mv[i] = 8 + val
                                else:
                                    indices_mv[i] = 9 + val

                indices_dict[zone] = indices

            elif zone in {"zone1", "zone2", "zone3", "zone4"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs[start_legs] + val

                indices_dict[zone] = indices
                start_legs += 1

            elif zone == "zone11":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 53:
                                    indices_mv[i] = 5 + val
                                elif 53 <= m < 141:
                                    indices_mv[i] = 6 + val
                                elif 141 <= m < 249:
                                    indices_mv[i] = 7 + val
                                elif 249 <= m < 351:
                                    indices_mv[i] = 8 + val
                                elif 351 <= m < 459:
                                    indices_mv[i] = 9 + val
                                elif 459 <= m < 547:
                                    indices_mv[i] = 4 + val
                                else:
                                    indices_mv[i] = 5 + val

                indices_dict[zone] = indices

            elif zone in {"zone12", "zone13", "zone14", "zone15"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs2[start_legs2] + val

                indices_dict[zone] = indices
                start_legs2 += 1

            else:
                raise NotImplementedError(f"zones not included in {ZONES} can not be allowed now.")

        return indices_dict

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef dict _indices2(self):
        cdef:
            dict indices_dict = {}
            str zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [13, 12, 11, 10]
            int[4] indices_legs2 = [13, 12, 11, 10]
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self._zones:

            with h5py.File(self._hdf5_path, mode="r") as h5file:
                ds_grids = h5file[self._grid_group][zone]["grids"]
                L = ds_grids.attrs["L"]
                M = ds_grids.attrs["M"]
                N = ds_grids.attrs["N"]
                NUM_CELLS = ds_grids.attrs["num_cells"]

            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 48:
                                    indices_mv[i] = 6 + val
                                elif 48 <= m < 170:
                                    indices_mv[i] = 5 + val
                                elif 170 <= m < 300:
                                    indices_mv[i] = 4 + val
                                elif 300 <= m < 430:
                                    indices_mv[i] = 9 + val
                                elif 430 <= m < 552:
                                    indices_mv[i] = 8 + val
                                else:
                                    indices_mv[i] = 7 + val

                indices_dict[zone] = indices

            elif zone in {"zone1", "zone2", "zone3", "zone4"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs[start_legs] + val

                indices_dict[zone] = indices
                start_legs += 1

            elif zone == "zone11":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11

                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 53:
                                    indices_mv[i] = 5 + val
                                elif 53 <= m < 141:
                                    indices_mv[i] = 4 + val
                                elif 141 <= m < 249:
                                    indices_mv[i] = 9 + val
                                elif 249 <= m < 351:
                                    indices_mv[i] = 8 + val
                                elif 351 <= m < 459:
                                    indices_mv[i] = 7 + val
                                elif 459 <= m < 547:
                                    indices_mv[i] = 6 + val
                                else:
                                    indices_mv[i] = 5 + val

                indices_dict[zone] = indices

            elif zone in {"zone12", "zone13", "zone14", "zone15"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs2[start_legs2] + val

                indices_dict[zone] = indices
                start_legs2 += 1

            else:
                raise NotImplementedError(f"zones not included in {ZONES} can not be allowed now.")

        return indices_dict

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef dict _indices3(self):
        cdef:
            dict indices_dict = {}
            str zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [12, 13, 10, 11]
            int[4] indices_legs2 = [12, 13, 10, 11]
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self._zones:

            with h5py.File(self._hdf5_path, mode="r") as h5file:
                ds_grids = h5file[self._grid_group][zone]["grids"]
                L = ds_grids.attrs["L"]
                M = ds_grids.attrs["M"]
                N = ds_grids.attrs["N"]
                NUM_CELLS = ds_grids.attrs["num_cells"]

            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 48:
                                    indices_mv[i] = 7 + val
                                elif 48 <= m < 170:
                                    indices_mv[i] = 8 + val
                                elif 170 <= m < 300:
                                    indices_mv[i] = 9 + val
                                elif 300 <= m < 430:
                                    indices_mv[i] = 4 + val
                                elif 430 <= m < 552:
                                    indices_mv[i] = 5 + val
                                else:
                                    indices_mv[i] = 6 + val

                indices_dict[zone] = indices

            elif zone in {"zone1", "zone2", "zone3", "zone4"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs[start_legs] + val

                indices_dict[zone] = indices
                start_legs += 1

            elif zone == "zone11":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11

                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 53:
                                    indices_mv[i] = 8 + val
                                elif 53 <= m < 141:
                                    indices_mv[i] = 9 + val
                                elif 141 <= m < 249:
                                    indices_mv[i] = 4 + val
                                elif 249 <= m < 351:
                                    indices_mv[i] = 5 + val
                                elif 351 <= m < 459:
                                    indices_mv[i] = 6 + val
                                elif 459 <= m < 547:
                                    indices_mv[i] = 7 + val
                                else:
                                    indices_mv[i] = 8 + val

                indices_dict[zone] = indices

            elif zone in {"zone12", "zone13", "zone14", "zone15"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs2[start_legs2] + val

                indices_dict[zone] = indices
                start_legs2 += 1

            else:
                raise NotImplementedError(f"zones not included in {ZONES} can not be allowed now.")

        return indices_dict

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef dict _indices4(self):
        cdef:
            dict indices_dict = {}
            str zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [11, 10, 13, 12]
            int[4] indices_legs2 = [11, 10, 13, 12]
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self._zones:

            with h5py.File(self._hdf5_path, mode="r") as h5file:
                ds_grids = h5file[self._grid_group][zone]["grids"]
                L = ds_grids.attrs["L"]
                M = ds_grids.attrs["M"]
                N = ds_grids.attrs["N"]
                NUM_CELLS = ds_grids.attrs["num_cells"]

            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 48:
                                    indices_mv[i] = 9 + val
                                elif 48 <= m < 170:
                                    indices_mv[i] = 8 + val
                                elif 170 <= m < 300:
                                    indices_mv[i] = 7 + val
                                elif 300 <= m < 430:
                                    indices_mv[i] = 6 + val
                                elif 430 <= m < 552:
                                    indices_mv[i] = 5 + val
                                else:
                                    indices_mv[i] = 4 + val

                indices_dict[zone] = indices

            elif zone in {"zone1", "zone2", "zone3", "zone4"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs[start_legs] + val

                indices_dict[zone] = indices
                start_legs += 1

            elif zone == "zone11":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11

                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 5:
                                indices_mv[i] = 0 + val
                            elif 5 <= l < 9:
                                indices_mv[i] = 1 + val
                            elif 9 <= l < 13:
                                indices_mv[i] = 2 + val
                            elif 13 <= l < 17:
                                indices_mv[i] = 3 + val
                            else:  # l >= 17
                                if m < 53:
                                    indices_mv[i] = 8 + val
                                elif 53 <= m < 141:
                                    indices_mv[i] = 7 + val
                                elif 141 <= m < 249:
                                    indices_mv[i] = 6 + val
                                elif 249 <= m < 351:
                                    indices_mv[i] = 5 + val
                                elif 351 <= m < 459:
                                    indices_mv[i] = 4 + val
                                elif 459 <= m < 547:
                                    indices_mv[i] = 9 + val
                                else:
                                    indices_mv[i] = 8 + val

                indices_dict[zone] = indices

            elif zone in {"zone12", "zone13", "zone14", "zone15"}:
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4) + start_zone11
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            indices_mv[i] = indices_legs2[start_legs2] + val

                indices_dict[zone] = indices
                start_legs2 += 1

            else:
                raise NotImplementedError(f"zones not included in {ZONES} can not be allowed now.")

        return indices_dict
