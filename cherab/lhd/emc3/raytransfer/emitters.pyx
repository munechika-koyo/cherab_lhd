"""The following emitters and integrators are used in ray transfer objects.

Note that these emitters support other integrators as well, however high performance
with other integrators is not guaranteed.
"""
cimport cython
from cherab.tools.raytransfer.emitters cimport RayTransferIntegrator
from raysect.optical cimport AffineMatrix3D, Point3D, Primitive, Ray, Spectrum, Vector3D, World
from raysect.optical.material cimport InhomogeneousVolumeEmitter, VolumeIntegrator

from ..cython.intfunction cimport IntegerFunction3D, autowrap_intfunction3d

__all__ = ["Discrete3DMeshRayTransferIntegrator", "Discrete3DMeshRayTransferEmitter"]


cdef class Discrete3DMeshRayTransferIntegrator(RayTransferIntegrator):
    """Calculates the distances traveled by the ray through the voxel defined on a tetrahedral mesh
    in 3d coordinate system: :math:`(X, Y, Z)`.

    This integrator is used with the `Discrete3DMeshRayTransferEmitter` material class and
    the `.Discrete3DMesh` to calculate ray transfer matrices (geometry matrices).
    The value for each voxel is stored in respective bin of the spectral array.
    The distances traveled by the ray through the voxel is calculated
    approximately and the accuracy depends on the integration step.

    Parameters
    ----------
    step : float
        Integration step, by default 0.001.
    min_samples : int
        Number of minimum samples of integration, by default 2.
    """
    def __init__(self, double step=0.001, int min_samples=2):
        super().__init__(step=step, min_samples=min_samples)

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum integrate(
        self,
        Spectrum spectrum,
        World world,
        Ray ray,
        Primitive primitive,
        InhomogeneousVolumeEmitter material,
        Point3D start_point,
        Point3D end_point,
        AffineMatrix3D world_to_primitive,
        AffineMatrix3D primitive_to_world
    ):

        cdef:
            Point3D start, end
            Vector3D direction
            int isource_pre, isource_current, it, n
            double length, t, dt, x, y, z, res

        if not isinstance(material, Discrete3DMeshRayTransferEmitter):
            raise TypeError(
                'Only Discrete3DMeshRayTransferEmitter material is supported '
                'by Discrete3DMeshRayTransferIntegrator'
            )
        start = start_point  # start point in world coordinates
        end = end_point  # end point in world coordinates
        direction = start.vector_to(end)  # direction of integration
        length = direction.get_length()  # integration length
        if length < 0.1 * self._step:  # return if ray's path is too short
            return spectrum
        direction = direction.normalise()  # normalized direction
        # number of points along ray's trajectory
        n = max(self._min_samples, <int>(length / self._step))
        dt = length / n  # integration step
        # cython performs checks on attributes of external class,
        # so it's better to do the checks before the loop
        isource_current = -1
        isource_pre = -1
        res = 0
        for it in range(n):
            t = (it + 0.5) * dt
            x = start.x + direction.x * t  # x coordinates of the points
            y = start.y + direction.y * t  # y coordinates of the points
            z = start.z + direction.z * t  # z coordinates of the points
            isource_current = material.index_function(x, y, z)  # get geometry grid index
            if isource_current != isource_pre:  # we moved to the next cell
                if isource_pre > -1:
                    # writing results for the current source
                    spectrum.samples_mv[isource_pre] += res
                isource_pre = isource_current
                res = 0
            if isource_current > -1:
                res += dt
        if isource_current > -1:
            spectrum.samples_mv[isource_current] += res

        return spectrum


cdef class Discrete3DMeshRayTransferEmitter(InhomogeneousVolumeEmitter):
    """A unit emitter defined on a Discrete3DMesh class, which can be used
    to calculate ray transfer matrices (geometry matrices).

    Note that for performance reason there are no boundary checks in `emission_function()`,
    or in `Discrete3DMeshRayTransferIntegrator`,
    so this emitter must be placed inside a bounding box.

    Parameters
    ----------
    index_function : callable
        Callable objects taking 3 positional arguments :math:`(X, Y, Z)`.
    bins : int
        Number of bins for the spectral array, by default 0.
    integration_step : float, optional
        The length of line integration step, by default 0.01.
    integrator : :obj:`~raysect.optical.material.emitter.inhomogeneous.VolumeIntegrator`, optional
        Volume integrator, by default `.Discrete3DMeshRayTransferIntegrator(integration_step)`.

    Examples
    --------
    .. code-block:: python

        from numpy import hypot
        from raysect.optical import World, translate, Point3D
        from raysect.primitive import Cylinder
        from cherab.lhd.emc3.raytransfer import Discrete3DMeshRayTransferEmitter

        def index_func(x, y, z):
            if hypot(x, y) <= 1:
                return 0
            elif 1 < hypot(x, y) <= 2:
                return 1
            else:
                return -1  # must be set -1 outside meshes.

        world = World()
        bins = 2  # Note that bins must be same as the number of meshes.
                  # Here thinking of two meshes like bins.shape = (2, ).
        material = Discrete3DMeshRayTransferEmitter(index_func, bins, integration_step=0.001)
        eps = 1.e-6  # ray must never leave the grid when passing through the volume
        radius = 2.0 - eps
        height = 10.0
        cylinder = Cylinder(
            radius,
            height,
            material=material,
            parent=world,
            transform=translate(0, 0, -0.5 * height)
        )

        camera.spectral_bins = material.bins
        # ray transfer matrix will be calculated for 600.5 nm
        camera.min_wavelength = 600.
        camera.max_wavelength = 601.
    """

    cdef:
        readonly IntegerFunction3D index_function
        readonly int _bins

    def __init__(
        self,
        object index_function not None,
        int bins=0,
        double integration_step=0.01,
        VolumeIntegrator integrator=None
    ):

        integrator = integrator or Discrete3DMeshRayTransferIntegrator(step=integration_step)
        super().__init__(integrator=integrator)

        self.index_function = autowrap_intfunction3d(index_function)
        self._bins = bins

    @property
    def bins(self):
        """
        Number of raytransfer meshes which must not exceed the maximum of `index_function`.

        :rtype: int
        """
        return self._bins

    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    cpdef Spectrum emission_function(
        self, Point3D point,
        Vector3D direction,
        Spectrum spectrum,
        World world,
        Ray ray,
        Primitive primitive,
        AffineMatrix3D world_to_primitive,
        AffineMatrix3D primitive_to_world
    ):

        cdef:
            int isource

        # index of the light source in 3D mesh
        isource = self.index_function(point.x, point.y, point.z)
        if isource < 0:  # grid cell is not mapped to any light source
            return spectrum
        spectrum.samples_mv[isource] += 1.  # unit emissivity
        return spectrum
