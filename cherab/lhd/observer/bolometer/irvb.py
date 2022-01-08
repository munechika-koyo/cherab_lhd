import numpy as np

from raysect.core import Node, Primitive
from raysect.core.math import Point3D, Vector3D
from raysect.optical import Ray
from raysect.optical.observer import TargettedCCDArray
from raysect.optical.material import NullMaterial

from cherab.tools.observers import BolometerSlit, BolometerFoil


XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)


# ----------------------- #
# IRVB observer class
# ----------------------- #
class IRVBCamera(Node):
    """InfRad Video Bolometer class inherited from Node.
    This instance stands for IRVB camera comprised of slit and foil instances.
    A slit and foil are generated by BolometerSlit and TargettedCCDArray classes, respectively.

    Parameters
    ----------
    camera_geometry: :obj:`~raysect.core.scenegraph.primitive.Primitive`
        A Raysect primitive to supply as the box/aperture geometry.
    parent : :obj:`~raysect.core.scenegraph.node.Node`
        The parent node of this camera in the scenegraph, often an optical World object.
    transform : :obj:`~raysect.core.math.affinematrix.AffineMatrix3D`
        The relative coordinate transform of this bolometer camera relative to the parent.
    name : str, optional
        IRVB name.

    .. prompt:: python >>> auto

       >>> from raysect.optical import World
       >>> from cherab.lhd.observer import IRVBCamera
       >>>
       >>> world = World()
       >>> camera = IRVBCamera(name="MyBolometer", parent=world)
    """

    def __init__(self, camera_geometry=None, parent=None, transform=None, name="") -> None:

        super().__init__(parent=parent, transform=transform, name=name)

        self._foil_detector = None
        self._slit = None

        if camera_geometry is not None:
            if not isinstance(camera_geometry, Primitive):
                raise TypeError("camera_geometry must be a primitive")
            camera_geometry.parent = self
        self._camera_geometry = camera_geometry

    @property
    def slit(self):
        """
        :obj:`~cherab.tools.observers.bolometry.BolometerSlit`: BolometerSlit instances.
        """
        return self._slit

    @slit.setter
    def slit(self, value):

        if not isinstance(value, BolometerSlit):
            raise TypeError("The slit attribute must be a BolometerSlit instance.")

        if value.parent != self:
            value.parent = self

        self._slit = value

    @property
    def foil_detector(self):
        """
        :obj:`~raysect.optical.observer.imaging.TargettedCCDArray`: a TargettedCCDArray instance.
        """
        return self._foil_detector

    @foil_detector.setter
    def foil_detector(self, value):

        if not isinstance(value, TargettedCCDArray):
            raise TypeError("The foil_detector attribute must be a TargettedCCDArray instance.")

        if value.parent != self:
            value.parent = self

        self._foil_detector = value

    @property
    def pixels_as_foils(self):
        """
        :obj:`numpy.ndarray`: an array, the element which is a BolometerFoil's instance defined by
        regarding each pixel as a bolometer foil.
        """

        nx, ny = self.foil_detector.pixels
        width = self.foil_detector.width
        pixel_pitch = width / nx
        height = pixel_pitch * ny

        # Foil pixels are defined in the foil's local coordinate system
        foil_upper_right = Point3D(width * 0.5, height * 0.5, 0)
        pixels = []
        for x in range(nx):
            pixel_column = []
            for y in range(ny):
                pixel_centre = foil_upper_right - (x + 0.5) * XAXIS * pixel_pitch - (y + 0.5) * YAXIS * pixel_pitch
                pixel = BolometerFoil(
                    detector_id="IRVB pixel ({},{})".format(x + 1, y + 1),
                    centre_point=pixel_centre,
                    basis_x=XAXIS,
                    basis_y=YAXIS,
                    dx=pixel_pitch,
                    dy=pixel_pitch,
                    slit=self._slit,
                    accumulate=False,
                    parent=self.foil_detector,
                )
                pixel_column.append(pixel)
            pixels.append(pixel_column)
        return np.asarray(pixels, dtype="object")

    @property
    def sightline_rays(self):
        """
        :obj:`numpy.ndarray` of :obj:`~raysect.optical.Ray`: an array containing sightline rays, each of which
        starts from the centre of each pixel and passes through
        the centre of the slit
        """
        return np.asarray(
            [
                [
                    Ray(pixel.centre_point, pixel.centre_point.vector_to(self._slit.centre_point))
                    for pixel in pixel_column
                ]
                for pixel_column in self.pixels_as_foils
            ],
            dtype="object",
        )

    def observe(self) -> None:
        """
        Take an observation with this camera.
        Call observe() on a foil detector
        """

        self.foil_detector.observe()

    def plot_bolometer_geometry(self, fig=None, plot_pixel_rays={}, show_foil_xy_axes=True):
        """3D plotting of bolometer geometry using plotly module
        If you want to use this method, must install `Plotly <https://plotly.com/python/>`_ module.

        Parameters
        ----------
        fig : :obj:`plotly.graph_objs.Figure`, optional
            Figure object created by plotly, by default ``fig = go.Figure()`` if fig is None.
        plot_pixel_rays : dict, optional
            setting option of plotting rays, by default
            {"plot", False, "pixels", (0, 0), "num_rays", 50}
        show_foil_xy_axes : bool, optional
            whether or not to show the local foil x, y axis, by default True

        Returns
        -------
        :obj:`plotly.graph_objs.Figure`
            Figure objects include some traces.
        """

        try:
            import plotly.graph_objects as go
        except ImportError:
            print("must install plotly module.")
            return

        if fig:
            if not isinstance(fig, go.Figure):
                raise TypeError("The fig argument must be of type plotly.graph_objs.Figure")
        else:
            fig = go.Figure()

        if not isinstance(plot_pixel_rays, dict):
            raise TypeError("The plot_pixel_rays argument must be of type dict.")

        # set default plot_pixel_rays
        plot_pixel_rays.setdefault("plot", False)  # whether or not to show pixel rays
        plot_pixel_rays.setdefault("pixels", (0, 0))  # select pixel number
        plot_pixel_rays.setdefault("num_rays", 50)  # number of rays plotted

        # target slit
        corners = [
            self.slit.centre_point - 0.5 * (+self.slit.dx * self.slit.basis_x + self.slit.dy * self.slit.basis_y),
            self.slit.centre_point - 0.5 * (-self.slit.dx * self.slit.basis_x + self.slit.dy * self.slit.basis_y),
            self.slit.centre_point - 0.5 * (-self.slit.dx * self.slit.basis_x - self.slit.dy * self.slit.basis_y),
            self.slit.centre_point - 0.5 * (+self.slit.dx * self.slit.basis_x - self.slit.dy * self.slit.basis_y),
        ]
        corners = np.array([[*point] for point in corners])
        fig.add_trace(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], opacity=0.6, text="target"))

        # foil screen
        foil_centre_point = ORIGIN.transform(self.foil_detector.to_root())
        basis_x = XAXIS.transform(self.foil_detector.to_root())
        basis_y = YAXIS.transform(self.foil_detector.to_root())
        width = self.foil_detector.width
        pixel_pitch = self.foil_detector.width / self.foil_detector.pixels[0]
        height = pixel_pitch * self.foil_detector.pixels[1]

        corners = [
            foil_centre_point - 0.5 * (+width * basis_x + height * basis_y),
            foil_centre_point - 0.5 * (-width * basis_x + height * basis_y),
            foil_centre_point - 0.5 * (-width * basis_x - height * basis_y),
            foil_centre_point - 0.5 * (+width * basis_x - height * basis_y),
        ]
        corners = np.array([[*point] for point in corners])
        fig.add_trace(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], opacity=0.6, text="screen"))

        # camera box
        xaxis = XAXIS.transform(self.to_root())
        yaxis = YAXIS.transform(self.to_root())
        zaxis = ZAXIS.transform(self.to_root())
        inner_box = self._camera_geometry.primitive_a.primitive_b
        lower = inner_box.lower.transform(self.to_root())
        upper = inner_box.upper.transform(self.to_root())
        lower_to_upper = lower.vector_to(upper)
        box_width = abs(lower_to_upper.dot(xaxis))
        box_height = abs(lower_to_upper.dot(yaxis))
        box_depth = abs(lower_to_upper.dot(zaxis))
        vertices = [
            lower,
            lower + box_width * xaxis,
            lower + box_width * xaxis + box_height * yaxis,
            lower + box_height * yaxis,
            lower + box_depth * zaxis,
            lower + box_depth * zaxis + box_width * xaxis,
            upper,
            upper - box_width * xaxis,
        ]
        vertices = np.array([[*vertex] for vertex in vertices])

        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.2,
                color="#7d7d7d",
                flatshading=True,
            )
        )

        # plot rays
        if plot_pixel_rays["plot"]:
            pixels = plot_pixel_rays["pixels"]
            UpperRight = Point3D(width * 0.5 - pixels[0] * pixel_pitch, height * 0.5 - pixels[1] * pixel_pitch, 0)
            points = [
                Point3D(*UpperRight).transform(self.foil_detector.to_root()),
                Point3D(UpperRight.x - pixel_pitch, UpperRight.y, UpperRight.z).transform(self.foil_detector.to_root()),
                Point3D(UpperRight.x - pixel_pitch, UpperRight.y - pixel_pitch, UpperRight.z).transform(
                    self.foil_detector.to_root()
                ),
                Point3D(UpperRight.x, UpperRight.y - pixel_pitch, UpperRight.z).transform(self.foil_detector.to_root()),
            ]

            corners = np.array([[*point] for point in points])
            fig.add_trace(
                go.Mesh3d(
                    x=corners[:, 0],
                    y=corners[:, 1],
                    z=corners[:, 2],
                    opacity=0.6,
                    text="selected pixel",
                    color="#7fff00",
                )
            )

            ray_temp = Ray()
            rays = self.foil_detector._generate_rays(pixels[0], pixels[1], ray_temp, plot_pixel_rays["num_rays"])
            for ray in rays:

                origin = ray[0].origin.transform(self.foil_detector.to_root())
                origin_0 = origin.copy()
                direction = ray[0].direction.transform(self.foil_detector.to_root())

                while True:
                    # Find the next intersection point
                    intersection = self.parent.hit(Ray(origin, direction))

                    if intersection is None:
                        raise RuntimeError("No material intersection was found for this sightline.")

                    elif isinstance(intersection.primitive.material, NullMaterial):
                        # apply a small displacement to avoid infinite self collisions due to numerics
                        hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                        ray_displacement = pixel_pitch / 100
                        origin = hit_point + direction * ray_displacement
                        continue
                    else:
                        hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                        break

                line = np.array([[*origin_0], [*hit_point]])

                fig.add_trace(
                    go.Scatter3d(
                        mode="lines",
                        x=line[:, 0],
                        y=line[:, 1],
                        z=line[:, 2],
                        line=dict(color="#FF0033", width=2),
                        text="sampled_ray",
                        showlegend=False,
                    )
                )

        # axis vector
        if show_foil_xy_axes:
            slit_sensor_separation = abs(foil_centre_point.vector_to(self.slit.centre_point).dot(zaxis))
            centre = Point3D(0, 0, -slit_sensor_separation).transform(self.to_root())
            xaxis_vector = go.Scatter3d(
                x=[centre.x, centre.x + width * self.slit.basis_x.x],
                y=[centre.y, centre.y + width * self.slit.basis_x.y],
                z=[centre.z, centre.z + width * self.slit.basis_x.z],
                marker=dict(color="rgb(256, 0, 0)", size=2),
                line=dict(color="rgb(256, 0, 0)"),
                showlegend=False,
            )
            yaxis_vector = go.Scatter3d(
                x=[centre.x, centre.x + height * self.slit.basis_y.x],
                y=[centre.y, centre.y + height * self.slit.basis_y.y],
                z=[centre.z, centre.z + height * self.slit.basis_y.z],
                marker=dict(color="rgb(0, 256, 0)", size=2),
                line=dict(color="rgb(0, 256, 0)"),
                showlegend=False,
            )
            fig.add_trace(xaxis_vector)
            fig.add_trace(yaxis_vector)

        return fig
