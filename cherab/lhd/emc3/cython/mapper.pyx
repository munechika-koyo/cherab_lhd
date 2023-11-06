"""Module to offer mapping classes
"""
cimport cython
from .intfunction cimport is_callable, autowrap_intfunction3d
from numpy import array, int32

__all__ = ["Mapper", "IndexMapper"]


cdef class Mapper(Function3D):
    """Mapping data array to function retuning its index value.

    This instance is callable function returning the element of 1-D `.data` array, the index of which
    is given by a Index function defined in 3-D space.

    If the index function returns an integer which is out of bounds or negative,
    an instance returns a default value defined by `.default_value`.

    :param IntegerFunction3D index_func: callable returning a index integer.
    :param array-like data: An 1D array of data.
    :param double default_value: The value to return outside the data size limits, by default 0.0.
    """

    def __init__(self, object index_func not None, object data not None, double default_value=0.0):

        # use numpy arrays to store data internally
        data = array(data, dtype=float)

        # validate arguments
        if data.ndim != 1:
            raise ValueError("data array must be 1D.")

        if not is_callable(index_func):
            raise TypeError("This function is not callable.")

        # populate internal attributes
        self._data_mv = data
        self._index_func = autowrap_intfunction3d(index_func)
        self._default_value = default_value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            int index

        index = self._index_func(x, y, z)

        if index < 0 or self._data_mv.size - 1 < index:
            return self._default_value
        else:
            return self._data_mv[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef bint inside_grids(self, double x, double y, double z):
        """mask function returning True if Point (x, y, z) in any grids, otherwise False.
        """
        return self._index_func(x, y, z) > -1


cdef class IndexMapper(IntegerFunction3D):
    """Mapping integer array to function retuning its index value.

    This instance is callable function returning the element of 1-D `.indices` array,
    the index of which is given by a Index function defined in 3-D space.

    If the index function returns an integer which is out of bounds or negative,
    an instance returns a default value defined by `.default_value`.

    :param IntegerFunction3D index_func: callable returning an index value.
    :param array-like indices: An 1D array of indices.
    :param double default_value: The value to return outside limits, by default -1.
    """

    def __init__(self, object index_func not None, object indices not None, int default_value=-1):

        # use numpy arrays to store data internally
        indices = array(indices, dtype=int32)

        # validate arguments
        if indices.ndim != 1:
            raise ValueError("indices array must be 1D.")

        if not is_callable(index_func):
            raise TypeError("This function must be .")

        # populate internal attributes
        self._indices_mv = indices
        self._index_func = index_func
        self._default_value = default_value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:

        cdef:
            int index

        index = self._index_func(x, y, z)

        if index < 0 or self._indices_mv.size - 1 < index:
            return self._default_value
        else:
            return self._indices_mv[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef bint inside_grids(self, double x, double y, double z):
        """mask function returning True if Point (x, y, z) in any grids, otherwise False."""
        return self._index_func(x, y, z) > -1
