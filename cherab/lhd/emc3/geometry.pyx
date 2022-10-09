"""
Module to offer helper functions to generate interpolate 3D function
"""
import json
import os
import sys
import inspect
import numpy as np

from numpy cimport ndarray, uint32_t
from libc.limits cimport INT_MIN
from raysect.primitive.mesh cimport TetraMesh

from cherab.lhd.emc3.cython cimport Discrete3DMesh
from cherab.lhd.emc3.cython.intfunction cimport IntegerFunction3D
cimport cython
from cython.parallel cimport prange

if not hasattr(sys.modules[__name__], "__file__"):
    __file__ = inspect.getfile(inspect.currentframe())


__all__ = ["CellIndex", "PhysIndex", "TomographyZone"]


cdef:
    list ZONES

DIRPATH = os.path.join(os.path.dirname(__file__), "data", "grid-360")
ZONES = [
    "zone0", "zone1", "zone2", "zone3", "zone4",
    "zone11", "zone12", "zone13", "zone14", "zone15"
]


cdef class _IndexBase(IntegerFunction3D):
    """
    Base class for EMC3-EIRENE cell index function.
    This class populates an IntegerFunction3D instance which returns a cell index
    when :math:`(X, Y, Z)` arguments are given.
    In addition, this offers common methods for specific cell index class.
    :obj:`.create_indices` method should be overrided in subclasses.

    :params list[str] zones: list of zones, by default ``["zone0", ..., "zone4", "zone11", ... "zone15"]``
    :params str dir_path: directry path to grid_config.json and TetraMesh files stored, by default ``../data/grid-360``
    :params bool populate: whether or not to populate instances of Discrete3DMesh, by default True
    """
    cdef:
        list _zones
        unicode _dir_path
        list _interpolaters
        dict _grid_config
    
    def __init__(self, list zones=ZONES, unicode dir_path=DIRPATH, bint populate=True):

        cdef:
            int i
            dict grid_config
            dict indices1, indices2, indices3, indices4
            TetraMesh tetra

        # store parameters as attributes
        self._zones = zones
        self._dir_path = dir_path
        self._interpolaters = []

        # load grid config
        with open(os.path.join(dir_path, "grid_config.json"), "r") as file:
            self._grid_config = json.load(file)

        # construct interpolaters
        if populate:
            self.create_interpolate()

    @property
    def zones(self):
        """
        list[str]: list of zones
        """
        return self._zones

    @property
    def dir_path(self):
        """
        str: directry path to grid_config.json and TetraMesh files
        """
        return self._dir_path

    cpdef tuple create_indices(self):
        """
        Base method for creating indices array.
        Here all 4 indices are created and returned.

        :return: tuple containing indices dictionaries.
        :rtype: tuple[dict, dict, dict, dict]
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
        """
        create indices array and populate instances of :obj:`~raysect.primitive.mesh.TetraMesh`.
        These instances are stored in ``self._interpolaters``
        """
        # create 1D indices arrays
        indices1_dict, indices2_dict, indices3_dict, indices4_dict = self.create_indices()

        for i in range(len(self._zones)):
            # load TetraMesh
            tetra = TetraMesh.from_file(os.path.join(self._dir_path, f"tetra-{self._zones[i]}.rsm"))
            
            # create interplater with Discrete3DMesh
            self._interpolaters.append(
                Discrete3DMesh(
                    tetra,
                    indices1_dict[self._zones[i]],
                    indices2_dict[self._zones[i]],
                    indices3_dict[self._zones[i]],
                    indices4_dict[self._zones[i]]
                )
            )

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
    """
    EMC3-EIRENE cell index function defined in zone0-4 & zone11-15.
    This class is subclass of :obj:`._IndexBase` which returns a cell index
    when :math:`(X, Y, Z)` arguments are given.

    :params list[str] zones: list of zones
    :params str dir_path: directry path to TetraMesh file stored.
    :params bool populate: whether or not to populate instances of Discrete3DMesh, by default True
    """
    def __init__(self, list zones=ZONES, unicode dir_path=DIRPATH, bint populate=True):
        raise NotImplementedError
        # super().__init__(zones, dir_path, populate)

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # cpdef dict create_indices(self, dict grid_config):
    #     cdef:
    #         dict cell_indices = {}
    #         int start = 0
    #         int last
    #         unicode zone

    #     for zone in self.zones:
    #         last = grid_config[zone]["num_cells"] + start
    #         cell_indices[zone] = np.arange(start=start, stop=last, dtype=np.uint32)
    #         start = last

    #     return cell_indices

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # cpdef dict _create_mapping_array(self, dict cell_indices, dict grid_config):
    #     cdef:
    #         dict mapping_array = {}
    #         unicode zone
    #         int L, M, N

    #     for zone in self.zones:
    #         L = grid_config[zone]["L"]
    #         M = grid_config[zone]["M"]
    #         N = grid_config[zone]["N"]

    #         if zone in {"zone0", "zone11"}:
    #             mapping_array[zone] = self._cell_index2mapping_array(cell_indices[zone], L, M, N)
    #         elif zone in {"zone1", "zone3", "zone13"}:
    #             zone_next = "zone" + str(int(zone.split("zone")[1]) + 1)
    #             mapping_array[zone] = self._cell_index2mapping_array(cell_indices[zone_next], L, M, N)
    #         elif zone in {"zone2", "zone4", "zone14"}:
    #             zone_prev = "zone" + str(int(zone.split("zone")[1]) - 1)
    #             mapping_array[zone] = self._cell_index2mapping_array(cell_indices[zone_prev], L, M, N)
    #         elif zone == "zone12":
    #             zone_next = "zone15"
    #             mapping_array[zone] = self._cell_index2mapping_array(cell_indices[zone_next], L, M, N)
    #         elif zone == "zone15":
    #             zone_prev = "zone12"
    #             mapping_array[zone] = self._cell_index2mapping_array(cell_indices[zone_prev], L, M, N)

    #     return mapping_array

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # cdef ndarray[uint32_t, ndim=1] _cell_index2mapping_array(
    #     self,
    #     uint32_t[::1] cell_indices,
    #     int L,
    #     int M,
    #     int N
    # ):
    #     return np.ravel(
    #         np.flip(
    #             np.reshape(cell_indices, (N - 1, M - 1, L - 1)),
    #             axis=1
    #         )
    #     )


cdef class PhysIndex(_IndexBase):
    """
    EMC3-EIRENE-defined Physical Cell Index function.

    This class inherits :obj:`._IndexBase` and the populated instance returns a physical cell index
    when :math:`(X, Y, Z)` arguments are given.
    Physical cell indices must be stored in ``dir_path`` as a ``indices-zone*.npy`` file.

    :params list[str] zones: list of zones
    :params str dir_path: directry path to TetraMesh file stored.
    :params bool populate: whether or not to populate instances of Discrete3DMesh, by default True
    """
    def __init__(self, list zones=ZONES, unicode dir_path=DIRPATH, bint populate=True):
        super().__init__(zones, dir_path, populate)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple create_indices(self):
        """
        all 4 indices are created and returned.
        Each indices must be stored in ``dir_path`` as a ``indices-zone*.npy`` file.

        :return: tuple containing indices dictionaries.
        :rtype: tuple[dict, dict, dict, dict]
        """
        cdef:
            dict indices_dict = {}
            unicode zone

        for zone in self._zones:
            indices_dict[zone] = np.load(os.path.join(self.dir_path, f"indices-{zone}.npy"))

        return (indices_dict, indices_dict, indices_dict, indices_dict)


cdef class TomographyZone(_IndexBase):
    """
    EMC3-EIRENE-based Tmography Zones function.

    This class inherits :obj:`.CellIndex` and the populated instance returns a Tomography zone index
    when :math:`(X, Y, Z)` arguments are given.
    Total zone size is 252 = 14 (poloidal) x 18 (0-18 deg in toroidal).

    :params list[str] zones: list of zones
    :params str dir_path: directry path to TetraMesh file stored.
    :params bool populate: whether or not to populate instances of Discrete3DMesh, by default True
    """
    def __init__(self, list zones=ZONES, unicode dir_path=DIRPATH, bint populate=True):
        super().__init__(zones, dir_path, populate)

    cpdef tuple create_indices(self):
        """
        all 4 indices are created and returned.
        Each of indices is created in each private method.

        :return: tuple containing indices dictionaries.
        :rtype: tuple[dict, dict, dict, dict]
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
            dict grid_config = self._grid_config
            unicode zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [10, 11, 12, 13]  # zone1, 2, 3, 4
            int[4] indices_legs2 = [10, 11, 12, 13]  # zone12, 13, 14, 15
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self.zones:

            L = grid_config[zone]["L"]
            M = grid_config[zone]["M"]
            N = grid_config[zone]["N"]
            NUM_CELLS = grid_config[zone]["num_cells"]
            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
            dict grid_config = self._grid_config
            unicode zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [13, 12, 11, 10]
            int[4] indices_legs2 = [13, 12, 11, 10]
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self.zones:

            L = grid_config[zone]["L"]
            M = grid_config[zone]["M"]
            N = grid_config[zone]["N"]
            NUM_CELLS = grid_config[zone]["num_cells"]
            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
            dict grid_config = self._grid_config
            unicode zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [12, 13, 10, 11]
            int[4] indices_legs2 = [12, 13, 10, 11]
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self.zones:

            L = grid_config[zone]["L"]
            M = grid_config[zone]["M"]
            N = grid_config[zone]["N"]
            NUM_CELLS = grid_config[zone]["num_cells"]
            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
            dict grid_config = self._grid_config
            unicode zone
            int L, M, N, NUM_CELLS, l, m, n, i, val, start_zone11, start_legs, start_legs2
            int[4] indices_legs = [11, 10, 13, 12]
            int[4] indices_legs2 = [11, 10, 13, 12]
            ndarray[uint32_t, ndim=1] indices
            uint32_t[::1] indices_mv

        start_zone11 = 126
        start_legs = 0
        start_legs2 = 0

        for zone in self.zones:

            L = grid_config[zone]["L"]
            M = grid_config[zone]["M"]
            N = grid_config[zone]["N"]
            NUM_CELLS = grid_config[zone]["num_cells"]
            indices = np.zeros(NUM_CELLS, dtype=np.uint32)
            indices_mv = indices

            if zone == "zone0":
                for n in prange(N - 1, nogil=True):
                    val = 14 * (n // 4)
                    for m in range(M - 1):
                        for l in range(L - 1):
                            i = n * (L - 1) * (M - 1) + m * (L - 1) + l
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
                            if l < 4:
                                indices_mv[i] = 0 + val
                            elif 4 <= l < 8:
                                indices_mv[i] = 1 + val
                            elif 8 <= l < 12:
                                indices_mv[i] = 2 + val
                            elif 12 <= l < 16:
                                indices_mv[i] = 3 + val
                            else:  # l >= 16
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
