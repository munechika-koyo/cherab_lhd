"""Module to offer mapping classes."""

import numbers

cimport cython
from .intfunction cimport is_callable, autowrap_intfunction3d
from numpy import array, int32

__all__ = ["Mapper", "IndexMapper"]


cdef class Mapper(Function3D):
    """Mapping data array to function retuning its index value.

    This instance is callable function returning the element of 1-D `.data` array,
    the index of which is given by a Index function defined in 3-D space.

    If the index function returns an integer which is out of bounds or negative,
    an instance returns a default value defined by `.default_value`.

    Parameters
    ----------
    index_func : callable[[float, float, float], int]
        Callable returning a index integer.
    data : (N, ) array_like
        An 1D array of data.
    default_value : float, optional
        The value to return outside the data size limits, by default 0.0.
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

    def __getstate__(self):
        return (
            self._index_func,
            self._data_mv,
            self._default_value
        )

    def __setstate__(self, state):
        (
            self._index_func,
            self._data_mv,
            self._default_value
        ) = state

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    # operator overloading
    def __add__(self, object b):
        if isinstance(b, Mapper):
            # a() + b()
            return AddMapper(self, b)
        elif isinstance(b, numbers.Real):
            # a() + B -> B + a()
            return AddScalarMapper(<double> b, self)
        return NotImplemented

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
        """Mask function returning True if Point (x, y, z) in any grids, otherwise False.
        """
        return self._index_func(x, y, z) > -1


cdef class AddMapper(Mapper):
    """A Function3D class that implements the addition of the results of two Mapper objects:
    f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an
    __add__() call on a Function3D object.

    Parameters
    ----------
    mapper1 : Mapper
        A Mapper object
    mapper2 : Mapper
        A Mapper object
    """

    def __init__(self, object mapper1, object mapper2):
        self._mapper1 = mapper1
        self._mapper2 = mapper2
        self._index_func = mapper1._index_func + mapper2._index_func

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self._mapper1.evaluate(x, y, z) + self._mapper2.evaluate(x, y, z)


cdef class AddScalarMapper(Mapper):
    """A Mapper class that implements the addition of scalar and the result of a Mapper object:
    K + f()

    This class is not intended to be used directly, but rather returned as the result of an
    __add__() call on a Mapper object.

    Parameters
    ----------
    value : float
        A double value.
    mapper : Mapper
        A Mapper object.
    """

    def __init__(self, double value, Mapper mapper):
        self._value = value
        self._mapper = mapper
        self._index_func = mapper._index_func

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self._value + self._mapper.evaluate(x, y, z)


cdef class IndexMapper(IntegerFunction3D):
    """Mapping integer array to function retuning its index value.

    This instance is callable function returning the element of 1-D `.indices` array,
    the index of which is given by a Index function defined in 3-D space.

    If the index function returns an integer which is out of bounds or negative,
    an instance returns a default value defined by `.default_value`.

    Parameters
    ----------
    index_func : callable[[float, float, float], int]
        Callable returning a index integer.
    indices : (N, ) array_like
        An 1D array of indices.
    default_value : int, optional
        The value to return outside the indices size limits, by default -1.
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

    def __getstate__(self):
        return (
            self._index_func,
            self._indices_mv,
            self._default_value
        )

    def __setstate__(self, state):
        (
            self._index_func,
            self._indices_mv,
            self._default_value
        ) = state

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

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
        """Mask function returning True if Point (x, y, z) in any grids, otherwise False."""
        return self._index_func(x, y, z) > -1
