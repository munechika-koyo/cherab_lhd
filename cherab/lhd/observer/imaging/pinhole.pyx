"""Module to define `.PinholeCamera` class
"""
from raysect.optical cimport AffineMatrix3D, Point3D, Ray, Vector3D, new_point3d, translate
from raysect.optical.observer.imaging.ccd cimport CCDArray
from raysect.optical.observer.pipeline cimport RGBPipeline2D

__all__ = ["PinholeCamera"]


cdef class PinholeCamera(CCDArray):
    """Observer modeling an idealised CCD-like imaging sensor with pinhole.

    A ray triggered from each pixel always goes through the arbitrary pinhole point.
    The distance from pinhole and imaging sensor represents focal length.

    Parameters
    ----------
    pixels : tuple[int, int], optional
        A tuple of pixel dimensions for the camera, by default ``(512, 512)``.
    width : float, optional
        The CCD sensor x-width in metres, by default 35 mm.
    focal_length : float, optional
        The distance between pinhole and CCD sensor, by default 24 mm.
    pinhole_point : tuple[float, float], optional
        The position of pinhole point from the centre of CCD,
        which represents as (x, y) coordinate, by default ``(0, 0)``.
    pipelines : list[Pipeline], optional
        The list of pipelines that will process the spectrum measured
        at each pixel by the camera, by default ``[RGBPipeline2D()]``.
    **kwargs
        kwargs and properties from `~raysect.optical.observer.base.observer.Observer2D` and
        `~raysect.optical.observer.base.observer._ObserverBase`.
    """

    cdef:
        double _focal_length
        tuple _pinhole_point

    def __init__(
        self,
        pixels=(512, 512),
        width=0.035,
        focal_length=0.024,
        pinhole_point=(0, 0),
        pipelines=None,
        **kwargs
    ):
        # set initial values
        self.focal_length = focal_length
        self.pinhole_point = pinhole_point

        pipelines = pipelines or [RGBPipeline2D()]

        super().__init__(pixels=pixels, width=width, pipelines=pipelines, **kwargs)

    @property
    def focal_length(self):
        """
        Focal length in metres.

        :rtype: float
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        focal_length = value
        if focal_length <= 0:
            raise ValueError("Focal length must be greater than 0.")
        self._focal_length = focal_length

    @property
    def pinhole_point(self):
        """
        Pinhole point from the centre of CCD

        :rtype: tuple
        """
        return self._pinhole_point

    @pinhole_point.setter
    def pinhole_point(self, value):
        pinhole_point = tuple(value)
        if len(pinhole_point) != 2:
            raise ValueError(
                "pinhole_point must be a 2 element tuple defining the x and y position."
            )

        self._pinhole_point = pinhole_point

    cpdef list _generate_rays(self, int ix, int iy, Ray template, int ray_count):

        cdef:
            double pixel_x, pixel_y
            list origin_points, rays
            Point3D origin
            Vector3D direction
            Ray ray
            AffineMatrix3D pixel_to_local

        # generate pixel transform
        pixel_x = self.image_start_x - self.image_delta * (ix + 0.5)
        pixel_y = self.image_start_y - self.image_delta * (iy + 0.5)
        pixel_to_local = translate(pixel_x, pixel_y, 0.0)

        # generate origin points
        origin_points = self.point_sampler.samples(ray_count)

        # generate pinhole point in local space
        pinhole_point = new_point3d(
            self._pinhole_point[0], self._pinhole_point[1], self._focal_length
        )

        # assemble rays
        rays = []
        for origin in origin_points:

            # transform to local space from pixel space
            origin = origin.transform(pixel_to_local)

            # generate direction vector
            direction = origin.vector_to(pinhole_point).normalise()

            ray = template.copy(origin, direction)

            # non-physical camera, samples radiance directly
            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            rays.append((ray, direction.z))

        return rays
