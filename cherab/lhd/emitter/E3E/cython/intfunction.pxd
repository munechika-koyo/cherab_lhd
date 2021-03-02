# cython: language_level=3

from libc.limits cimport INT_MIN
from raysect.core.math.function.base cimport Function


cdef class IntegerFunction3D:
    cdef int evaluate(self, double x, double y, double z) except? INT_MIN

cdef inline bint is_callable(object f):
    """
    Tests if an object is a python callable or a Function3D object.

    :param object f: Object to test.
    :return: True if callable, False otherwise.
    """
    if isinstance(f, IntegerFunction3D):
        return True

    # other function classes are incompatible
    if isinstance(f, Function):
        return False

    return callable(f)