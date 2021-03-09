cimport numpy as np
from raysect.core.math.function.float cimport Function3D
from cherab.lhd.emitter.E3E.cython cimport IntegerFunction3D


cdef class EMC3Mapper(Function3D):

    cdef:
        np.ndarray _data
        double[::1] _data_mv
        double _default_value
        IntegerFunction3D _index_func


    cdef double evaluate(self, double x, double y, double z) except? -1e999