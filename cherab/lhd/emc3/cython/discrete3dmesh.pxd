from libc.limits cimport INT_MIN
from numpy cimport import_array, uint32_t
# from raysect.primitive.mesh cimport TetraMeshData
from .tetra_mesh cimport TetraMeshData

from .intfunction cimport IntegerFunction3D

import_array()


cdef class Discrete3DMesh(IntegerFunction3D):

    cdef:
        TetraMeshData _tetra_mesh
        uint32_t[::1] _indices1_mv
        uint32_t[::1] _indices2_mv
        uint32_t[::1] _indices3_mv
        uint32_t[::1] _indices4_mv

    cdef int evaluate(self, double x, double y, double z) except? INT_MIN
