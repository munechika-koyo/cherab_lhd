# cython: language_level=3

# This module is optimized for EMC3-EIRINE in LHD by Koyo Munechika.

from libc.limits cimport INT_MIN


cdef class IntegerFunction3D:
    """
    Cython optimised class for representing an arbitrary 3D function returning a integer.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y, double z):
        """ Evaluate the function f(x, y, z)

        :param float x: function parameter x
        :param float y: function parameter y
        :param float y: function parameter z
        :rtype: int
        """
        return self.evaluate(x, y, z)
    
    def __repr__(self):
        return 'IntegerFunction3D()'