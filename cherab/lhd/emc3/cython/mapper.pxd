from libc.limits cimport INT_MIN
from numpy cimport ndarray, import_array
from raysect.core.math.function.float cimport Function3D

from .intfunction cimport IntegerFunction3D

import_array()


cdef class EMC3Mapper(Function3D):

    cdef:
        double[::1] _data_mv
        double _default_value
        IntegerFunction3D _index_func


    cdef double evaluate(self, double x, double y, double z) except? -1e999

    cpdef int inside_grids(self, double x, double y, double z)


cdef class IndexMapper(IntegerFunction3D):

    cdef:
        int[::1] _indices_mv
        int _default_value
        IntegerFunction3D _index_func


    cdef int evaluate(self, double x, double y, double z) except? INT_MIN

    cpdef int inside_grids(self, double x, double y, double z)
