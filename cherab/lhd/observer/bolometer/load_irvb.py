"""Module offers helper functions related to a IRVB camera."""
from __future__ import annotations

import json
from importlib import resources

import numpy as np
from raysect.core.math import Point3D, Vector3D, rotate_basis, rotate_vector, rotate_z, translate
from raysect.optical.material import AbsorbingSurface
from raysect.optical.observer import TargettedCCDArray
from raysect.primitive import Box, Subtract

from cherab.tools.observers import BolometerSlit

from ..imaging.pinhole import PinholeCamera
from .irvb import IRVBCamera

__all__ = ["load_irvb", "load_irvb_as_pinhole_camera", "calcam_virtual_calibration"]

XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)


def load_irvb(
    port: str = "6.5U", flange: str = "CC01_04", parent: World | None = None
) -> IRVBCamera:
    """helper function of generating a IRVB camera object. An IRVB's
    configuration is defined in json file, the name of which is "IRVB.json"
    stored in ``../data folder``.

    Parameters
    ----------
    port
        user-specified port name, by default ``"6.5U"``
    flange
        specific flange name, by default ``"CC01_04"``
    parent
        The parent node of this camera in the scenegraph, often
        an optical :obj:`~raysect.core.scenegraph.world.World` object, by default None

    Returns
    -------
    :obj:`.IRVBCamera`
        populated :obj:`.IRVBCamera` instance
    """

    # import IRVB configs as a resource
    with resources.open_text("cherab.lhd.observer.bolometer.data", "IRVB.json") as file:
        raw_data = json.load(file)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][flange]
    except KeyError as err:
        raise KeyError(f"spcified parameters: {port}-{flange} are not defined.") from err

    # Construct Foil and Slit from port local coordinates
    if raw_data.get("basis_x"):
        SLIT_WIDTH, SLIT_HEIGHT = raw_data["slit_size"]
        FOIL_WIDTH, FOIL_HEIGHT = raw_data["foil_size"]
        foil_centre = Point3D(*raw_data["foil_centre"])
        slit_centre_in_local = Point3D(*raw_data["slit_centre_in_local"])
        basis_x = Vector3D(*raw_data["basis_x"]).normalise()
        basis_y = Vector3D(*raw_data["basis_y"]).normalise()
        PIXELS = raw_data["pixels"]

        BOX_WIDTH = FOIL_WIDTH + 1.0e-3
        BOX_HEIGHT = FOIL_HEIGHT + 1.0e-3
        foil_forward = basis_x.cross(basis_y)
        BOX_DEPTH = slit_centre_in_local.z

    # ----------------------------------- #
    # Construct bolometer camera object
    # ----------------------------------- #

    # Camera Box
    inner_box = Box(
        lower=Point3D(-BOX_WIDTH * 0.5, -BOX_HEIGHT * 0.5, -1e-3),
        upper=Point3D(BOX_WIDTH * 0.5, BOX_HEIGHT * 0.5, BOX_DEPTH),
    )
    outside_box = Box(
        lower=inner_box.lower - Vector3D(1e-5, 1e-5, 1e-5),
        upper=inner_box.upper + Vector3D(1e-5, 1e-5, 1e-5),
    )
    camera_box = Subtract(outside_box, inner_box)

    aperture = Box(
        lower=Point3D(
            slit_centre_in_local.x - SLIT_WIDTH * 0.5,
            slit_centre_in_local.y - SLIT_HEIGHT * 0.5,
            slit_centre_in_local.z - 1e-4,
        ),
        upper=Point3D(
            slit_centre_in_local.x + SLIT_WIDTH * 0.5,
            slit_centre_in_local.y + SLIT_HEIGHT * 0.5,
            slit_centre_in_local.z + 1e-4,
        ),
    )

    camera_box = Subtract(camera_box, aperture, name="camera_box")
    camera_box.material = AbsorbingSurface()

    bolometer_camera = IRVBCamera(
        camera_geometry=camera_box, parent=parent, name=f"IRVB-{port}-{flange}"
    )

    # Slit
    slit = BolometerSlit(
        slit_id="slit",
        centre_point=slit_centre_in_local,
        basis_x=basis_x,
        dx=SLIT_WIDTH,
        basis_y=basis_y,
        dy=SLIT_HEIGHT,
        parent=bolometer_camera,
    )

    # Foil
    foil = TargettedCCDArray(
        [slit.target],
        pixels=PIXELS,
        width=FOIL_WIDTH,
        parent=bolometer_camera,
        targetted_path_prob=0.99,
        name="foil",
    )

    # apply a slit & foil
    bolometer_camera.slit = slit
    bolometer_camera.foil_detector = foil

    # camera rotate
    bolometer_camera.transform = (
        rotate_z(raw_data["rotate"]) * translate(*foil_centre) * rotate_basis(foil_forward, basis_y)
    )

    return bolometer_camera


def load_irvb_as_pinhole_camera(
    port: str = "6.5U", flange: str = "BC02", parent: World | None = None
) -> PinholeCamera:
    """helper function of generating an IRVB camera as Pinhole Camara An IRVB's
    configuration is defined in json file, the name of which is "IRVB.json"
    stored in ``../data folder``.

    Parameters
    ----------
    port
        user-specified port name, by default "6.5U"
    flange
        The variant of IRVB model, by default "BC02"
    parent
        The parent node of this camera in the scenegraph, often
        an optical :obj:`~raysect.core.scenegraph.world.World` object, by default None

    Returns
    -------
    :obj:`.PinholeCamera`
        populated :obj:`.PinholeCamera` instance
    """
    # import IRVB configs as a resource
    with resources.open_text("cherab.lhd.observer.bolometer.data", "IRVB.json") as file:
        raw_data = json.load(file)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][flange]
    except KeyError as err:
        raise KeyError(f"spcified parameters: {port} or {flange} are not defined.") from err

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
    slit_centre, slit_basis_x, slit_basis_y, _, _ = slit_geometry
    foil_centre, foil_basis_x, foil_basis_y, _, _ = foil_geometry

    # vector the centre of foil to that of slit
    foil_normal = foil_basis_x.cross(foil_basis_y)
    vector_foil_to_slit_centre = foil_centre.vector_to(slit_centre)

    # distance between slit and foil
    if foil_normal.dot(vector_foil_to_slit_centre) < 0:
        foil_normal = -1.0 * foil_normal
    SLIT_FOIL_SEPARATION = abs(foil_normal.dot(foil_corners[0].vector_to(slit_corners[0])))

    # foil  dimensions
    PIXELS = raw_data["pixels"]
    PIXEL_PITCH = min([foil_geometry[i + 3] / PIXELS[i] for i in range(2)])
    SCREEN_WIDTH = PIXEL_PITCH * PIXELS[0]
    # SCREEN_HEIGHT = PIXEL_PITCH * PIXELS[1]

    # slit local coords.
    slit_centre_local = Point3D(
        -vector_foil_to_slit_centre.dot(slit_basis_x),
        vector_foil_to_slit_centre.dot(slit_basis_y),
        0,
    )

    # ----------------------------------- #
    # Construct Pinhole Camera object
    # ----------------------------------- #

    bolometer_camera = PinholeCamera(
        pixels=PIXELS,
        width=SCREEN_WIDTH,
        focal_length=SLIT_FOIL_SEPARATION,
        pinhole_point=(slit_centre_local.x, slit_centre_local.y),
        parent=parent,
        name=f"IRVB-{port}-{flange}",
    )

    # camera rotate
    bolometer_camera.transform = (
        rotate_z(raw_data["rotate"])
        * translate(*slit_centre)
        * rotate_basis(foil_normal, foil_basis_y)
        * translate(-slit_centre_local.x, -slit_centre_local.y, 0)
    )

    return bolometer_camera


def calcam_virtual_calibration(
    port: str = "6.5U", flange: str = "BC02"
) -> dict[str, float | tuple[int, int] | tuple[float, float, float]]:
    """generate virtual calibration parameters used in calcam for IRVB. This
    function calculate virtual pixel pitch, camera position, and camera
    orientation (target and roll)

    Parameters
    ----------
    port
        user-specified port name, by default "6.5U"
    flange
        The variant of IRVB model, by default "BC02"

    Returns
    -------
    dict
        combination of each key and value is as follows:
        "pixel_pitch": float,
        "pixels": (int, int),
        "focal_length": float,
        "camera_pos": (float, float, float),
        "target": (float, float, float),
        "roll": float
    """
    # import IRVB configs as a resource
    with resources.open_text("cherab.lhd.observer.bolometer.data", "IRVB.json") as file:
        raw_data = json.load(file)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][flange]
    except KeyError as err:
        raise KeyError(f"spcified parameters: {port} or {flange} are not defined.") from err

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
    slit_centre, _, _, _, _ = slit_geometry
    foil_centre, foil_basis_x, foil_basis_y, _, _ = foil_geometry

    # vector the centre of foil to that of slit
    foil_normal = foil_basis_x.cross(foil_basis_y)
    vector_foil_to_slit_centre = foil_centre.vector_to(slit_centre)

    # distance between slit and foil
    if foil_normal.dot(vector_foil_to_slit_centre) < 0:
        foil_normal = -1.0 * foil_normal
    SLIT_FOIL_SEPARATION = abs(foil_normal.dot(foil_corners[0].vector_to(slit_corners[0])))

    # calculate virtual angle of view
    vec_a = foil_corners[0].vector_to(slit_centre).normalise()
    vec_b = foil_corners[2].vector_to(slit_centre).normalise()
    angle = vec_a.angle(vec_b)

    # foil  dimensions
    PIXELS: tuple[int, int] = tuple(raw_data["pixels"])
    PIXEL_PITCH = 2.0 * SLIT_FOIL_SEPARATION * np.tan(0.5 * np.deg2rad(angle)) / np.hypot(*PIXELS)

    # camera position
    camera_pos = slit_centre.transform(rotate_z(raw_data["rotate"]))
    target = (vec_a + vec_b).transform(rotate_z(raw_data["rotate"])) + Vector3D(
        camera_pos.x, camera_pos.y, camera_pos.z
    )
    # camera roll
    theta = foil_corners[0].vector_to(foil_corners[2]).angle(foil_basis_y)
    up = (vec_a - vec_b).transform(
        rotate_z(raw_data["rotate"]) * rotate_vector(theta, vec_a + vec_b)
    )

    roll = np.rad2deg(np.arctan2(up.y, up.x))

    return dict(
        pixel_pitch=PIXEL_PITCH,
        pixels=PIXELS,
        focal_length=SLIT_FOIL_SEPARATION,
        camera_pos=(camera_pos.x, camera_pos.y, camera_pos.z),
        target=(target.x, target.y, target.z),
        roll=roll,
    )


def _centre_basis_and_dimensions(
    corners: Point3D,
) -> tuple[Point3D, Vector3D, Vector3D, float, float]:
    """Calculate the centre point, basis vectors, width and height given 4
    corners."""
    centre = Point3D(
        np.mean([corner.x for corner in corners]),
        np.mean([corner.y for corner in corners]),
        np.mean([corner.z for corner in corners]),
    )
    basis_x = corners[0].vector_to(corners[1]).normalise()
    basis_y = corners[1].vector_to(corners[2]).normalise()
    width = corners[0].distance_to(corners[1])
    height = corners[1].distance_to(corners[2])
    return (centre, basis_x, basis_y, width, height)


def _corners_coords(
    centre: Point3D, basis_x: Vector3D, basis_y: Vector3D, width: float, height: float
) -> list[Point3D]:
    """Calculate the rectangular corners."""
    return [
        centre + basis_x * width * 0.5 + basis_y * height * 0.5,
        centre + basis_x * width * 0.5 - basis_y * height * 0.5,
        centre - basis_x * width * 0.5 - basis_y * height * 0.5,
        centre - basis_x * width * 0.5 + basis_y * height * 0.5,
    ]


if __name__ == "__main__":
    from raysect.optical import World

    world = World()
    irvb = load_irvb(port="6.5L", parent=world)

    print("debug")
