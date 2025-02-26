"""Module to offer cythonized Integer Functions for EMC3-EIRENE in LHD."""

import numbers
from libc.limits cimport INT_MIN
from raysect.core.math.function.base cimport Function

__all__ = ["IntegerFunction3D", "IntegerConstant3D", "PythonIntegerFunction3D"]


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

        Parameters
        ----------
        x : float
            Function parameter x.
        y : float
            Function parameter y.
        z : float
            Function parameter z.

        Returns
        -------
        int
            The result of the function f(x, y, z).
        """
        return self.evaluate(x, y, z)

    def __repr__(self):
        return 'IntegerFunction3D()'

    def __add__(self, object b):
        if is_callable(b):
            # a() + b()
            return AddFunction3D(self, b)
        elif isinstance(b, int):
            # a() + B -> B + a()
            return AddScalar3D(<int> b, self)
        return NotImplemented

    def __radd__(self, object a):
        return self.__add__(a)


cdef class AddFunction3D(IntegerFunction3D):
    """
    A IntegerFunction3D class that implements the addition of the results of two IntegerFunction3D
    objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an
    __add__() call on a IntegerFunction3D object.

    Parameters
    ----------
    function1 : IntegerFunction3D
        The first IntegerFunction3D object to be evaluated.
    function2 : IntegerFunction3D
        The second IntegerFunction3D object to be evaluated.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_intfunction3d(function1)
        self._function2 = autowrap_intfunction3d(function2)

    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:
        return self._function1.evaluate(x, y, z) + self._function2.evaluate(x, y, z)


cdef class AddScalar3D(IntegerFunction3D):
    """
    A IntegerFunction3D class that implements the addition of scalar and the result of a
    IntegerFunction3D object: K + f()

    This class is not intended to be used directly, but rather returned as the result of an
    __add__() call on a IntegerFunction3D object.

    Parameters
    ----------
    value : int
        The scalar value to be added to the result of the function.
    function : IntegerFunction3D
        The IntegerFunction3D object to be evaluated.
    """

    def __init__(self, int value, object function):
        self._value = value
        self._function = autowrap_intfunction3d(function)

    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:
        return self._value + self._function.evaluate(x, y, z)


cdef class IntegerConstant3D(IntegerFunction3D):
    """
    Wraps a scalar constant with a IntegerFunction3D object.

    This class allows a numeric Python scalar, such as an integer, to
    interact with cython code that requires a IntegerFunction3D object. The scalar must
    be convertible to integer. The value of the scalar constant will be returned
    independent of the arguments the function is called with.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a IntegerFunction3D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of IntegerFunction3D. If the object is not a
    IntegerFunction3D object it should be wrapped using this class for internal use.

    Parameters
    ----------
    value : int
        The constant value to be returned by the function.


    See Also
    --------
    autowrap_intfunction3d
    """
    def __init__(self, int value):
        self._value = value

    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:
        return self._value


cdef class PythonIntegerFunction3D(IntegerFunction3D):
    """
    Wraps a python callable object with a IntegerFunction3D object.

    This class allows a python object to interact with cython code that requires
    a IntegerFunction3D object. The python object must implement __call__() expecting
    three arguments.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a IntegerFunction3D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of IntegerFunction3D. If the object is not a
    IntegerFunction3D object it should be wrapped using this class for internal use.

    See Also
    --------
    autowrap_intfunction3d
    """

    def __init__(self, object function):
        self.function = function

    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:
        return self.function(x, y, z)


cdef IntegerFunction3D autowrap_intfunction3d(object obj):
    """
    Automatically wraps the supplied python object in a PythonIntegerFunction3D
    or IntegerConstant3D object.

    If this function is passed a valid IntegerFunction3D object, then the IntegerFunction3D
    object is simply returned without wrapping.

    If this function is passed a numerical scalar (int or float), a Constant3D
    object is returned.

    This convenience function is provided to simplify the handling of Function3D
    and python callable objects in constructors, functions and setters.
    """

    if isinstance(obj, IntegerFunction3D):
        return <IntegerFunction3D> obj
    elif isinstance(obj, Function):
        raise TypeError('A IntegerFunction3D object is required.')
    elif isinstance(obj, numbers.Integral):
        return IntegerConstant3D(obj)
    else:
        return PythonIntegerFunction3D(obj)


def _autowrap_intfunction3d(obj):
    """Expose cython function for testing."""
    return autowrap_intfunction3d(obj)
