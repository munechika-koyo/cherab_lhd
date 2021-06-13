import os
from json import load
import numpy as np
from raysect.core.math import Point3D, Vector3D, translate, rotate_basis, rotate_z
from raysect.primitive import Box, Subtract
from raysect.optical.observer import TargettedCCDArray
from raysect.optical.material import AbsorbingSurface
from cherab.tools.observers import BolometerSlit
from cherab.lhd.observer.bolometer import IRVBCamera


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")

XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)


def load_irvb(file=None, parent=None):
    """helper function of generating a IRVB camera object
    using LHD setting from .json file.

    Parameters
    ----------
    file : str
        json file name written
    parent : [type], optional
        [description], by default None

    Returns
    -------
    IRVBCamera
        IRVBCamera instance
    """

    # default json file
    file = file or os.path.join(DATA_DIRECTORY, "IB65U.json")

    # load IRVB geometry information from json file
    with open(file=file, mode="r") as f:
        raw_data = load(f)

    # transform slit and foil corners to Point3D data
    slit_corners = np.asarray_chkfinite(raw_data["slit"])
    foil_corners = np.asarray_chkfinite(raw_data["foil"])

    slit_corners = [Point3D(*slit_corners[i, :]) for i in range(4)]
    foil_corners = [Point3D(*foil_corners[i, :]) for i in range(4)]

    slit_geometry = _centre_basis_and_dimensions(slit_corners)
    foil_geometry = _centre_basis_and_dimensions(foil_corners)

    # -------------------------- #
    # define geometry constants
    # -------------------------- #
    slit_centre, slit_basis_x, slit_basis_y, slit_width, slit_height = slit_geometry
    foil_centre, foil_basis_x, foil_basis_y, _, _ = foil_geometry

    foil_normal = foil_basis_x.cross(foil_basis_y)
    if foil_normal.dot(foil_centre.vector_to(slit_centre)) < 0:
        foil_normal = -1.0 * foil_normal
    SLIT_SENSOR_SEPARATION = abs(foil_normal.dot(foil_corners[0].vector_to(slit_corners[0])))

    # IRVB camera box dimensions
    BOX_HEIGHT = foil_geometry[4] + 1e-3  # add padding
    BOX_WIDTH = foil_geometry[3] + 1e-3
    BOX_DEPTH = SLIT_SENSOR_SEPARATION + 1e-3

    # slit dimensions
    SLIT_HEIGHT = slit_geometry[4]
    SLIT_WIDTH = slit_geometry[3]

    # foil screen dimensions
    PIXELS = raw_data["pixels"]
    PIXEL_PITCH = min([foil_geometry[i + 3] / PIXELS[i] for i in range(2)])
    SCREEN_WIDTH = PIXEL_PITCH * PIXELS[0]
    # SCREEN_HEIGHT = PIXEL_PITCH * PIXELS[1]

    # slit local coords.
    vector_foil_to_slit_centre = foil_centre.vector_to(slit_centre)
    centre_slit_local = Point3D(
        -vector_foil_to_slit_centre.dot(slit_basis_x), vector_foil_to_slit_centre.dot(slit_basis_y), 0
    )

    # ----------------------------------- #
    # Construct bolometer camera object
    # ----------------------------------- #

    # Camera Box
    inner_box = Box(
        lower=Point3D(-BOX_WIDTH * 0.5, -BOX_HEIGHT * 0.5, -BOX_DEPTH),
        upper=Point3D(BOX_WIDTH * 0.5, BOX_HEIGHT * 0.5, 0),
    )
    outside_box = Box(
        lower=inner_box.lower - Vector3D(1e-5, 1e-5, 1e-5), upper=inner_box.upper + Vector3D(1e-5, 1e-5, 1e-5)
    )
    camera_box = Subtract(outside_box, inner_box)

    aperture = Box(
        lower=Point3D(centre_slit_local.x - SLIT_WIDTH * 0.5, centre_slit_local.y - SLIT_HEIGHT * 0.5, -1e-4),
        upper=Point3D(centre_slit_local.x + SLIT_WIDTH * 0.5, centre_slit_local.y + SLIT_HEIGHT * 0.5, 1e-4),
    )

    camera_box = Subtract(camera_box, aperture, name="camera_box")
    camera_box.material = AbsorbingSurface()

    bolometer_camera = IRVBCamera(camera_geometry=camera_box, parent=parent, name=raw_data["name"])

    # Slit
    slit = BolometerSlit(
        slit_id="slit",
        centre_point=centre_slit_local,
        basis_x=XAXIS,
        dx=SLIT_WIDTH,
        basis_y=YAXIS,
        dy=SLIT_HEIGHT,
        parent=bolometer_camera,
    )

    # Screen
    screen = TargettedCCDArray(
        [slit.target],
        pixels=PIXELS,
        width=SCREEN_WIDTH,
        parent=bolometer_camera,
        transform=translate(0, 0, -SLIT_SENSOR_SEPARATION),
        targetted_path_prob=0.99,
        name="foil",
    )

    # apply a slit & screen to a IRVBcamera foil_detector attibute
    bolometer_camera.slit = slit
    bolometer_camera.foil_detector = screen

    # camera rotate
    bolometer_camera.transform = (
        rotate_z(raw_data["rotate"])
        * translate(*slit_centre)
        * rotate_basis(foil_normal, foil_basis_y)
        * translate(-centre_slit_local.x, -centre_slit_local.y, 0)
    )

    return bolometer_camera


def _centre_basis_and_dimensions(corners):
    """Calculate the centre point, basis vectors, width and height given 4 corners."""
    centre = Point3D(
        np.mean([corner.x for corner in corners]),
        np.mean([corner.y for corner in corners]),
        np.mean([corner.z for corner in corners]),
    )
    basis_x = corners[0].vector_to(corners[1]).normalise()
    basis_y = corners[1].vector_to(corners[2]).normalise()
    width = corners[0].distance_to(corners[1])
    height = corners[1].distance_to(corners[2])
    return centre, basis_x, basis_y, width, height


if __name__ == "__main__":
    from raysect.optical import World

    world = World()
    irvb = load_irvb(parent=world)

    print("debug")
