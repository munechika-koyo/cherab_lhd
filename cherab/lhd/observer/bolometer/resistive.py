import numpy as np

from raysect.core import translate, rotate_basis, Point3D, Vector3D, Ray as CoreRay, World
from raysect.core.math.sampler import TargettedHemisphereSampler, RectangleSampler3D
from raysect.optical.observer import SightLine, TargettedPixel
from raysect.optical.material.material import NullMaterial

from cherab.tools.observers import BolometerSlit, BolometerCamera
from cherab.tools.observers.bolometry import mask_corners

R_2_PI = 1 / (2 * np.pi)


class BolometerFoil(TargettedPixel):
    """
    A rectangular foil bolometer detector.

    When instantiating a detector, the position and orientation
    (i.e. centre_point, basis_x and basis_y) are given in the local coordinate
    system of the foil's parent, usually a :obj:`~cherab.tools.observers.bolometry.BolometerCamera` instance.
    When these properties are accessed after instantiation, they are given in
    the coordinate system of the world.

    Parameters
    ----------
    detector_id : str
        The name for this detector.
    centre_point : :obj:`~raysect.core.math.point.Point3D`
        The centre point of the detector.
    basis_x : :obj:`~raysect.core.math.vector.Vector3D`
        The x basis vector for the detector.
    dx : float
        The width of the detector along the x basis vector.
    basis_y : :obj:`~raysect.core.math.vector.Vector3D`
        The y basis vector for the detector.
    dy : float
        The height of the detector along the y basis vector.
    accumulate: bool
        Whether this observer should accumulate samples
        with multiple calls to observe.
    curvature_radius: float
        Detectors in real bolometer cameras typically
        have curved corners due to machining limitations. This parameter species
        the corner radius.
    **kwargs : :obj:`~raysect.optical.observer.nonimaging.targetted_pixel.TargettedPixel` properties, optional
        *kwargs* are used to specify properties like a parent, transform, pipelines, etc.


    Attributes
    ----------
    normal_vector : :obj:`~raysect.core.math.vector.Vector3D`
        The normal vector of the detector constructed from
        the cross product of the x and y basis vectors.
    sightline_vector : :obj:`~raysect.core.math.vector.Vector3D`
        The vector that points from the centre of the foil
        detector to the centre of the slit. Defines the effective sightline vector of the
        detector.

    Examples
    --------
    .. prompt:: python >>> auto

       >>> from raysect.core import Point3D, Vector3D
       >>> from raysect.optical import World
       >>> from cherab.lhd.observer import BolometerFoil
       >>>
       >>> world = World()
       >>>
       >>> # construct basis vectors
       >>> basis_x = Vector3D(1, 0, 0)
       >>> basis_y = Vector3D(0, 1, 0)
       >>> basis_z = Vector3D(0, 0, 1)
       >>>
       >>> # specify a detector, you need already created slit and camera objects
       >>> dx = 0.0025
       >>> dy = 0.005
       >>> centre_point = Point3D(0, 0, -0.08)
       >>> detector = BolometerFoil("ch#1", centre_point, basis_x, dx, basis_y, dy, slit, parent=camera)
    """

    def __init__(self, detector_id, centre_point, basis_x, dx, basis_y, dy, slit,
                 accumulate=False, curvature_radius=0, **kwargs):

        # perform validation of input parameters

        if not isinstance(dx, (float, int)):
            raise TypeError("dx argument for BolometerFoil must be of type float/int.")
        if not dx > 0:
            raise ValueError("dx argument for BolometerFoil must be greater than zero.")

        if not isinstance(dy, (float, int)):
            raise TypeError("dy argument for BolometerFoil must be of type float/int.")
        if not dy > 0:
            raise ValueError("dy argument for BolometerFoil must be greater than zero.")

        if not isinstance(slit, BolometerSlit):
            raise TypeError("slit argument for BolometerFoil must be of type BolometerSlit.")

        if not isinstance(centre_point, Point3D):
            raise TypeError("centre_point argument for BolometerFoil must be of type Point3D.")

        if not isinstance(curvature_radius, (float, int)):
            raise TypeError("curvature_radius argument for BolometerFoil "
                            "must be of type float/int.")
        if curvature_radius < 0:
            raise ValueError("curvature_radius argument for BolometerFoil "
                             "must not be negative.")

        if not isinstance(basis_x, Vector3D):
            raise TypeError("The basis vectors of BolometerFoil must be of type Vector3D.")
        if not isinstance(basis_y, Vector3D):
            raise TypeError("The basis vectors of BolometerFoil must be of type Vector3D.")

        basis_x = basis_x.normalise()
        basis_y = basis_y.normalise()
        normal_vec = basis_x.cross(basis_y)
        self._slit = slit
        self._curvature_radius = curvature_radius
        self._accumulate = accumulate

        # setup root bolometer foil transform
        translation = translate(centre_point.x, centre_point.y, centre_point.z)
        rotation = rotate_basis(normal_vec, basis_y)

        super().__init__([slit.target], targetted_path_prob=1.0,
                         pixel_samples=1000, x_width=dx, y_width=dy, spectral_bins=1, quiet=True,
                         transform=translation * rotation, name=detector_id, **kwargs)

        # round off the detector corners, if applicable
        if self._curvature_radius > 0:
            mask_corners(self)

    def __repr__(self):
        """Returns a string representation of this BolometerFoil object."""
        return "<BolometerFoil - " + self.name + ">"

    @property
    def centre_point(self):
        return Point3D(0, 0, 0).transform(self.to_root())

    @property
    def normal_vector(self):
        return Vector3D(0, 0, 1).transform(self.to_root())

    @property
    def basis_x(self):
        return Vector3D(1, 0, 0).transform(self.to_root())

    @property
    def basis_y(self):
        return Vector3D(0, 1, 0).transform(self.to_root())

    @property
    def sightline_vector(self):
        return self.centre_point.vector_to(self._slit.centre_point)

    @property
    def slit(self):
        return self._slit

    @property
    def curvature_radius(self):
        return self._curvature_radius

    @property
    def accumulate(self):
        return self._accumulate

    @accumulate.setter
    def accumulate(self, value):
        for pipeline in self.pipelines:
            pipeline.accumulate = value
            # Discard any samples from previous accumulate behaviour
            pipeline.value.clear()

    def as_sightline(self):
        """
        Constructs a SightLine observer for this bolometer.

        Returns
        -------
        :obj:`~raysect.optical.observer.nonimaging.sightline.SightLine`
        """

        los_observer = SightLine(pipelines=self.pipelines, pixel_samples=1, quiet=True,
                                 parent=self, name=self.name)
        los_observer.render_engine = self.render_engine
        los_observer.spectral_bins = self.spectral_bins
        los_observer.min_wavelength = self.min_wavelength
        los_observer.max_wavelength = self.max_wavelength
        # The observer's Z axis should be aligned along the line of sight vector
        los_observer.transform = rotate_basis(
            self.sightline_vector.transform(self.to_local()), self.basis_y
        )

        return los_observer

    def trace_sightline(self):
        """
        Traces the central sightline through the detector to see where the sightline terminates.
        Raises a RuntimeError exception if no intersection was found.

        Returns
        --------
        tuple
            A tuple containing the origin point, hit point and terminating surface primitive.
        """

        if not isinstance(self.root, World):
            raise ValueError("This BolometerFoil is not connected to a valid World scenegraph object.")

        origin = self.centre_point
        direction = self.sightline_vector

        while True:

            # Find the next intersection point of the ray with the world
            intersection = self.root.hit(CoreRay(origin, direction))

            if intersection is None:
                raise RuntimeError("No material intersection was found for this sightline.")

            elif isinstance(intersection.primitive.material, NullMaterial):
                # apply a small displacement to avoid infinite self collisions due to numerics
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                ray_displacement = min(self.x_width, self.y_width) / 100
                origin = hit_point + direction * ray_displacement
                continue

            else:
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                return self.centre_point, hit_point, intersection.primitive

    def calculate_etendue(self, ray_count=10000, batches=10, max_distance=1e999):
        """
        Calculates the etendue of this detector.

        This function calculates the detectors etendue by evaluating the fraction of rays that
        pass un-impeded through the detector's aperture.

        Parameters
        ----------
        ray_count : int
            The number of rays used per batch.
        batches : int
            The number of batches used to estimate the error on the etendue calculation.
        max_distance : float
            The maximum distance from the detector to consider intersections.
            If a ray makes it further than this, it is assumed to have passed through the aperture,
            regardless of what it hits. Use this if there are other primitives present in the scene
            which do not form the aperture.

        Returns
        --------
        tuple
            tuple of (etendue, etendue_error).
        """

        if batches < 5:
            raise ValueError("We enforce a minimum batch size of 5 to ensure reasonable statistics.")

        target = self.slit.target

        world = self.slit.root
        detector_transform = self.to_root()

        # generate bounding sphere and convert to local coordinate system
        sphere = target.bounding_sphere()
        spheres = [(sphere.centre.transform(self.to_local()), sphere.radius, 1.0)]
        # instance targetted pixel sampler to sample directions
        targetted_sampler = TargettedHemisphereSampler(spheres)
        # instance rectangle pixel sampler to sample origins
        point_sampler = RectangleSampler3D(width=self.x_width, height=self.y_width)

        def etendue_single_run(_):
            """Worker function to calculate the etendue: will be run <batches> times"""
            origins = point_sampler(samples=ray_count)
            passed = 0.0
            for origin in origins:
                # obtain targetted vector sample
                direction, pdf = targetted_sampler(origin, pdf=True)
                path_weight = R_2_PI * direction.z / pdf
                # Transform to world space
                origin = origin.transform(detector_transform)
                direction = direction.transform(detector_transform)
                while True:
                    # Find the next intersection point of the ray with the world
                    intersection = world.hit(CoreRay(origin, direction, max_distance))
                    if intersection is None:
                        passed += 1 * path_weight
                        break
                    if isinstance(intersection.primitive.material, NullMaterial):
                        hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                        # apply a small displacement to avoid infinite self collisions due to numerics
                        ray_displacement = min(self.x_width, self.y_width) / 100
                        origin = hit_point + direction * ray_displacement
                        continue
                    break
            etendue = (passed / ray_count) * self.sensitivity
            return etendue

        etendues = []
        self.render_engine.run(list(range(batches)), etendue_single_run, etendues.append)
        etendue = np.mean(etendues)
        etendue_error = np.std(etendues)

        return etendue, etendue_error


class BolometerCamera(BolometerCamera):
    """
    A group of bolometer sight-lines under a single scenegraph node.

    This class that manages a collection of :obj:`.BolometerFoil`
    objects. Allows combined observation and display control simultaneously.
    Some attributes override :obj:`~cherab.tools.observers.bolometery.BolometerCamera`.

    Parameters
    ----------
    camera_geometry : :obj:`~raysect.core.scenegraph.primitive.Primitive`
        A Raysect primitive to supply as the box/aperture geometry.
    name : str
        The name for this bolometer camera.
    **kwargs : :obj:`~cherab.tools.observers.bolometery.BolometerCamera` properties, optional
        *kwargs* are used to specify properties like a parent, transform, etc.

    Attributes
    ----------
    foil_detectors : list
        A list of the foil detector objects that belong to this camera.
    slits : list
        A list of the bolometer slit objects that belong to this camera.

    Examples
    --------
    .. prompt:: python >>> auto

       >>> from raysect.optical import World
       >>> from cherab.lhd.observer import BolometerCamera
       >>>
       >>> world = World()
       >>> camera = BolometerCamera(name="MyBolometer", parent=world)
    """

    def __init__(self, camera_geometry=None, parent=None, transform=None, name=''):

        super().__init__(camera_geometry=camera_geometry, parent=parent, transform=transform, name=name)

    @property
    def foil_detectors(self):
        return self._foil_detectors.copy()

    @foil_detectors.setter
    def foil_detectors(self, value):

        if not isinstance(value, list):
            raise TypeError(
                "The foil_detectors attribute of BolometerCamera must be a list of "
                "BolometerFoils."
            )

        # Prevent external changes being made to this list
        value = value.copy()
        for foil_detector in value:
            if not isinstance(foil_detector, BolometerFoil):
                raise TypeError(
                    f"The foil_detectors attribute of BolometerCamera must be a list of "
                    f"BolometerFoil objects. Value {foil_detector} is not a BolometerFoil."
                )
            if foil_detector.slit not in self._slits:
                self._slits.append(foil_detector.slit)
            foil_detector.parent = self

        self._foil_detectors = value

    def add_foil_detector(self, foil_detector):
        """
        Add the given detector to this camera.

        Parameters
        ----------
        foil_detector : BolometerFoil
            An instanced bolometer foil detector.

        .. prompt:: python >>> auto

           >>> bolometer_camera.add_foil_detector(foil_detector)
        """

        if not isinstance(foil_detector, BolometerFoil):
            raise TypeError(
                "The foil_detector argument must be of type BolometerFoil."
            )

        if foil_detector.slit not in self._slits:
            self._slits.append(foil_detector.slit)

        foil_detector.parent = self
        self._foil_detectors.append(foil_detector)

    def observe(self):
        """
        Take an observation with this camera.

        Calls observe() on each foil detector and returns their power measurements.

        Returns
        -------
        list
            list of pipelines, each of which belongs to each foil_detector
        """

        foil_pipelines = []
        for foil_detector in self._foil_detectors:
            foil_detector.observe()
            foil_pipelines.append(foil_detector.pipelines)

        return foil_pipelines
