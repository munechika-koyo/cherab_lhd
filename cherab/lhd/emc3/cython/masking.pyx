"""Module to offer masking features EMC3-EIRENE in LHD."""

cimport cython
from libc.limits cimport INT_MIN


cdef class Mask(IntegerFunction3D):
    """Masking EMC3-EIRINE grid space to identify whether or not a mesh exists at the point.

    This instance is callable function returning 1 corresponding in
    3D space where EMC3's index function returns a physical index, otherwise 0.

    Parameters
    ----------
    index_func : callable[[float, float, float], int]
        EMC3's index_funcion returning a physical index.
    """

    def __init__(self, object index_func not None):

        # validate arguments
        if not callable(index_func):
            raise TypeError("This function is not callable.")

        # populate internal attributes
        self._index_func = index_func

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:

        cdef:
            int index

        index = self._index_func(x, y, z)

        if index < 0:
            return 0
        else:
            return 1
