from libc.limits cimport INT_MIN

from .intfunction cimport IntegerFunction3D


cdef class EMC3Mask(IntegerFunction3D):

    cdef:
        IntegerFunction3D _index_func


    cdef int evaluate(self, double x, double y, double z) except? INT_MIN
