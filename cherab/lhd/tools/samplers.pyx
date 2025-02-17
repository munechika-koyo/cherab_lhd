"""This module provides a set of sampling functions for rapidly generating samples of a 3D functions
with cylindrical coords.

These functions use C calls when sampling Function3D objects and are therefore considerably faster
than the equivalent Python code.
"""
from numpy import cos, empty, linspace, sin

cimport cython
from libc.math cimport M_PI
from numpy cimport float64_t, ndarray
from raysect.core.math.function.float cimport Function2D, Function3D, autowrap_function3d
from cherab.core.math.samplers cimport sample2d

__all__ = ["sample3d_rz", "sample_xy_plane"]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sample3d_rz(object function3d, tuple r_range, tuple z_range, double phi=0.0):
    """Sample a 3D function with a specified toroidal angle and a certain range of R-Z coordinates

    Parameters
    ----------
    function3d : Callable[[double, double, double], double]
        Python function or `Function3D` object.
    r_range : tuple[double, double, int]
        R-axis sampling range: :math:`(R_\\text{min}, R_\\text{max}, N)`, where :math:`N` is the
        number of samples.
    z_range : tuple[double, double, int]
        Z-axis sampling range: :math:`(Z_\\text{min}, Z_\\text{max}, M)`, where :math:`M` is the
        number of samples.
    phi : double, optional
        Toroidal angle in degree, by default 0.0 [deg].

    Returns
    -------
    r_points : (N, ) array_like
        R-axis sample points.
    z_points : (M, ) array_like
        Z-axis sample points.
    function_samples : (N, M) array_like
        Sampled function values.

    Examples
    --------
    >>> def f1(x, y, z):
    >>>     return x**3 + y**2 + z
    >>>
    >>> r_pts, z_pts, f_vals = sample3d_rz(f1, (1, 3, 3), (1, 3, 3), 0.0)
    >>> r_pts
    array([1., 2., 3.])
    >>> f_vals
    array([[ 2.,  3.,  4.],
           [ 9., 10., 11.],
           [28., 29., 30.]])
    """

    cdef:
        int i, j
        Function3D f3d
        int r_samples, z_samples
        double phi_rad
        ndarray[float64_t, ndim=1] r, x, y, z
        ndarray[float64_t, ndim=2] v
        float64_t[::1] x_view, y_view, z_view
        float64_t[:, ::1] v_view

    if len(r_range) != 3:
        raise ValueError(
            "R range must be a tuple containing: (min range, max range, no. of samples)."
        )

    if len(z_range) != 3:
        raise ValueError(
            "Z range must be a tuple containing: (min range, max range, no. of samples)."
        )

    if r_range[0] > r_range[1]:
        raise ValueError("Minimum r range can not be greater than maximum r range.")

    if z_range[0] > z_range[1]:
        raise ValueError("Minimum z range can not be greater than maximum z range.")

    if r_range[2] < 1:
        raise ValueError("The number of r samples must be >= 1.")

    if z_range[2] < 1:
        raise ValueError("The number of z samples must be >= 1.")

    phi_rad = phi * M_PI / 180.0

    f3d = autowrap_function3d(function3d)
    r_samples = r_range[2]
    z_samples = z_range[2]

    r = linspace(r_range[0], r_range[1], r_samples)
    z = linspace(z_range[0], z_range[1], z_samples)
    x = linspace(r_range[0] * cos(phi_rad), r_range[1] * cos(phi_rad), r_samples)
    y = linspace(r_range[0] * sin(phi_rad), r_range[1] * sin(phi_rad), r_samples)
    v = empty((r_samples, z_samples))

    # obtain memoryviews for fast, direct memory access
    x_view = x
    y_view = y
    z_view = z
    v_view = v

    for i in range(r_samples):
        for j in range(z_samples):
            v_view[i, j] = f3d.evaluate(x_view[i], y_view[i], z_view[j])

    return (r, z, v)


cdef class XYplane(Function2D):
    """Wrapper class for a 3D function to allow it to be sampled in the X-Y plane.

    This class is not intended to be used directly, use the `.sample_xy_plane` function instead.

    Parameters
    ----------
    func : callable
        A Python function or Function3D object.
    z : float
        Z-axis value to evaluate the function, by default 0.0.
    """

    cdef:
        Function3D func
        double z

    def __init__(self, object func, double z = 0.0):
        self.func = autowrap_function3d(func)
        self.z = z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, double x, double y) except? -1e999:
        return self.func.evaluate(x, y, self.z)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple[ndarray, ndarray, ndarray] sample_xy_plane(
    object func,
    tuple x_range,
    tuple y_range,
    double z = 0.0,
):
    """Sample a 3D function over the specified range with X-Y coords at a certain Z-axis value.

    Parameters
    ----------
    func : Callable[[double, double, double], double]
        Python function or `Function3D` object.
    x_range : tuple[double, double, int]
        X-axis sampling range: :math:`(X_\\text{min}, X_\\text{max}, N)`, where :math:`N` is the
        number of samples.
    y_range : tuple[double, double, int]
        Y-axis sampling range: :math:`(Y_\\text{min}, Y_\\text{max}, M)`, where :math:`M` is the
        number of samples.
    z : double, optional
        Z-axis value to evaluate the function, by default 0.0.

    Returns
    -------
    x_points : (N, ) array_like
        X-axis sample points.
    y_points : (M, ) array_like
        Y-axis sample points.
    function_samples : (N, M) array_like
        Sampled function values.

    Examples
    --------
    >>> def f1(x, y, z):
    >>>     return x**3 + y**2 + z
    >>>
    >>> x_pts, y_pts, f_vals = sample_xy_plane(f1, (1, 3, 3), (1, 3, 3), 0.0)
    >>> x_pts
    array([1., 2., 3.])
    >>> y_pts
    array([1., 2., 3.])
    >>> f_vals
    array([[ 2.,  5., 10.],
           [ 9., 12., 17.],
           [28., 31., 36.]])
    """

    cdef:
        ndarray x_pts, y_pts, samples

    x_pts, y_pts, samples = sample2d(XYplane(func, z=z), x_range, y_range)

    return x_pts, y_pts, samples
