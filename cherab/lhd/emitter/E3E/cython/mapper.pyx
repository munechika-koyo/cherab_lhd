
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

    @classmethod
    def instance(cls, EMC3Mapper instance not None, object data not None, object default_value=None):
        """
        Creates a new EMC3Mapper instance from an existing EMC3Mapper instance.

        The new EMC3Mapper instance will share the same internal acceleration
        data as the original EMC3Mapper. The data and default_value
        settings of the new instance can be redefined by setting the appropriate
        attributes. If any of the attributes are set to None (default) then the
        value from the original EMC3Mapper will be copied.

        This method should be used if the user has multiple sets of data
        that lie on the same EMC3 index function. Using this methods avoids the
        repeated rebuilding of index function by sharing the
        geometry data between multiple EMC3Mapper objects.

        :param EMC3Mapper instance: EMC3Mapper object.
        :param array-like data: An 1D array of data defined by EMC3.
        :param double default_value: The value to return outside the data size limits, by default 0.0.
        :return: An EMC3Mapper object.
        :rtype: EMC3Mapper
        """
        cdef EMC3Mapper m

        # copy source data
        m = EMC3Mapper.__new__(EMC3Mapper)
        m._index_func = instance._index_func

        data = array(data, dtype=float64)
        if data.ndim != 1:
            raise ValueError("data array must be 1D.")
        m._data = data

        # create memoryview
        m._data_mv = data

        # do we have a replacement default value?
        if default_value is None:
            m._default_value = instance._default_value
        else:
            m._default_value = default_value
        
        return m

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