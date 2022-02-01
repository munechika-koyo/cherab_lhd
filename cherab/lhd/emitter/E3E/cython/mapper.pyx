
cimport cython
from numpy import array, float64


cdef class EMC3Mapper(Function3D):
    """
    Mapping EMC3-Eirine data to tetrahedra meshs.
    Several EMC3's data are stored in some EMC3's geometric cells.
    This instance is callable function returning physical data corresponding in 
    3D space where EMC3's index function returns a physical index

    :param IntegerFunction3D index_func: EMC3's index_funcion returning a physical index.
    :param array-like data: An 1D array of data defined by EMC3.
    :param double default_value: The value to return outside the data size limits, by default 0.0.
    """

    def __init__(self, object index_func not None, object data not None, double default_value=0.0):

        # use numpy arrays to store data internally
        data = array(data, dtype=float64)

        # validate arguments
        if data.ndim != 1:
            raise ValueError("data array must be 1D.")

        if not callable(index_func):
            raise TypeError("This function is not callable.")

        # populate internal attributes
        self._data = data
        self._data_mv = data
        self._index_func = index_func
        self._default_value = default_value


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        
        cdef:
            int index

        index = self._index_func(x, y, z)

        if index < 0 or self._data.size - 1 < index:
            return self._default_value
        else:
            return self._data_mv[index]
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef int inside_grids(self, double x, double y, double z):
        """
        mask function returning True if Point (x, y, z) in any grids,
        otherwise False.
        """
        return self._index_func(x, y, z) > 0
