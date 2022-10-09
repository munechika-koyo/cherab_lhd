from libc.limits cimport INT_MIN
from numpy cimport ndarray
from raysect.core.math.function.float cimport Function3D
from cherab.lhd.emc3.cython.intfunction cimport IntegerFunction3D


cdef class EMC3Mapper(Function3D):

    cdef:
        ndarray _data
        double[::1] _data_mv
        double _default_value
        IntegerFunction3D _index_func


    cdef double evaluate(self, double x, double y, double z) except? -1e999

    cpdef int inside_grids(self, double x, double y, double z)


cdef class IndexMapper(IntegerFunction3D):

    cdef:
        ndarray _indices
        int[::1] _indices_mv
        int _default_value
        IntegerFunction3D _index_func


    cdef int evaluate(self, double x, double y, double z) except? INT_MIN

    cpdef int inside_grids(self, double x, double y, double z)