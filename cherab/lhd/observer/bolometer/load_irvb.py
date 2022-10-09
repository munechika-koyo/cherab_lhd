"""
Module offers helper functions related to a IRVB camera.
"""
from __future__ import annotations

import os
from json import load

import numpy as np
from cherab.lhd.observer.bolometer import IRVBCamera
from cherab.lhd.observer.imaging import PinholeCamera
from cherab.tools.observers import BolometerSlit
from raysect.core.math import Point3D, Vector3D, rotate_basis, rotate_vector, rotate_z, translate
from raysect.optical.material import AbsorbingSurface
from raysect.optical.observer import TargettedCCDArray
from raysect.primitive import Box, Subtract

__all__ = ["load_irvb", "load_irvb_as_pinhole_camera", "calcam_virtual_calibration"]


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")

XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)


def load_irvb(
    port: str = "6.5U", model_variant: str = "BC02", parent: World | None = None
) -> IRVBCamera:
    """helper function of generating a IRVB camera object.
    An IRVB's configuration is defined in json file, the name of which is "IRVB.json"
    stored in ``../data folder``.

    Parameters
    ----------
    port
        user-specified port name, by default "6.5U"
    model_variant
        The variant of IRVB model, by default "BC02"
    parent
        The parent node of this camera in the scenegraph, often
        an optical :obj:`~raysect.core.scenegraph.world.World` object, by default None

    Returns
    -------
    :obj:`.IRVBCamera`
        populated :obj:`.IRVBCamera` instance
    """

    # Load IRVB config. from .json file
    file = os.path.join(DATA_DIRECTORY, "IRVB.json")
    with open(file=file, mode="r") as f:
        raw_data = load(f)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][model_variant]
    except KeyError:
        raise KeyError(f"spcified parameters: {port} or {model_variant} are not defined.")

    # generate slit and foil geometry
    foil_corners = [Point3D(*xyz) for xyz in raw_data["foil"]]
    foil_geometry = _centre_basis_and_dimensions(foil_corners)
    foil_centre, foil_basis_x, foil_basis_y, _, _ = foil_geometry

    if "slit" in raw_data:
        slit_corners = [Point3D(*xyz) for xyz in raw_data["slit"]]
        slit_geometry = _centre_basis_and_dimensions(slit_corners)

    elif "slit_centre" in raw_data:
        slit_centre = Point3D(*raw_data["slit_centre"])
        SLIT_WIDTH, SLIT_HEIGHT = raw_data["slit_size"]
        slit_geometry = (slit_centre, foil_basis_x, foil_basis_y, SLIT_WIDTH, SLIT_HEIGHT)
        slit_corners = _corners_coords(
            slit_centre, foil_basis_x, foil_basis_y, SLIT_WIDTH, SLIT_HEIGHT
        )
    else:
        raise KeyError("slit or slit_centre must be exist in IRVB database.")

    slit_centre, slit_basis_x, slit_basis_y, SLIT_WIDTH, SLIT_HEIGHT = slit_geometry

    # vector the centre of foil to that of slit
    foil_normal = foil_basis_x.cross(foil_basis_y)
    vector_foil_to_slit_centre = foil_centre.vector_to(slit_centre)

    # distance between slit and foil
    if foil_normal.dot(vector_foil_to_slit_centre) < 0:
        foil_normal = -1.0 * foil_normal  # normal in the direction from foil to slit
    SLIT_FOIL_SEPARATION: float = abs(foil_normal.dot(foil_corners[0].vector_to(slit_corners[0])))

    # IRVB camera box dimensions
    BOX_HEIGHT = foil_geometry[4] + 1e-3  # add padding
    BOX_WIDTH = foil_geometry[3] + 1e-3
    BOX_DEPTH = SLIT_FOIL_SEPARATION + 1e-3

    # foil screen dimensions
    PIXELS = raw_data["pixels"]
    PIXEL_PITCH: float = min([foil_geometry[i + 3] / PIXELS[i] for i in range(2)])
    SCREEN_WIDTH: float = PIXEL_PITCH * PIXELS[0]
    # SCREEN_HEIGHT = PIXEL_PITCH * PIXELS[1]

    # slit local coords.
    centre_slit_local = Point3D(
        -vector_foil_to_slit_centre.dot(slit_basis_x),
        vector_foil_to_slit_centre.dot(slit_basis_y),
        0,
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
        lower=inner_box.lower - Vector3D(1e-5, 1e-5, 1e-5),
        upper=inner_box.upper + Vector3D(1e-5, 1e-5, 1e-5),
    )
    camera_box = Subtract(outside_box, inner_box)

    aperture = Box(
        lower=Point3D(
            centre_slit_local.x - SLIT_WIDTH * 0.5, centre_slit_local.y - SLIT_HEIGHT * 0.5, -1e-4
        ),
        upper=Point3D(
            centre_slit_local.x + SLIT_WIDTH * 0.5, centre_slit_local.y + SLIT_HEIGHT * 0.5, 1e-4
        ),
    )

    camera_box = Subtract(camera_box, aperture, name="camera_box")
    camera_box.material = AbsorbingSurface()

    bolometer_camera = IRVBCamera(
        camera_geometry=camera_box, parent=parent, name=f"IRVB-{port}-{model_variant}"
    )

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
        transform=translate(0, 0, -SLIT_FOIL_SEPARATION),
        targetted_path_prob=0.99,
        name="foil",
    )

    # apply a slit & screen
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


def load_irvb_as_pinhole_camera(
    port: str = "6.5U", model_variant: str = "BC02", parent: World | None = None
) -> PinholeCamera:
    """helper function of generating an IRVB camera as Pinhole Camara
    An IRVB's configuration is defined in json file, the name of which is "IRVB.json"
    stored in ``../data folder``.

    Parameters
    ----------
    port
        user-specified port name, by default "6.5U"
    model_variant
        The variant of IRVB model, by default "BC02"
    parent
        The parent node of this camera in the scenegraph, often
        an optical :obj:`~raysect.core.scenegraph.world.World` object, by default None

    Returns
    -------
    :obj:`.PinholeCamera`
        populated :obj:`.PinholeCamera` instance
    """

    # Load IRVB config. from .json file
    file = os.path.join(DATA_DIRECTORY, "IRVB.json")
    with open(file=file, mode="r") as f:
        raw_data = load(f)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][model_variant]
    except KeyError:
        raise KeyError(f"spcified parameters: {port} or {model_variant} are not defined.")

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

    # foil screen dimensions
    PIXELS = raw_data["pixels"]
    PIXEL_PITCH = min([foil_geometry[i + 3] / PIXELS[i] for i in range(2)])
    SCREEN_WIDTH = PIXEL_PITCH * PIXELS[0]
    # SCREEN_HEIGHT = PIXEL_PITCH * PIXELS[1]

    # slit local coords.
    centre_slit_local = Point3D(
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
        pinhole_point=(centre_slit_local.x, centre_slit_local.y),
        parent=parent,
        name=f"IRVB-{port}-{model_variant}",
    )

    # camera rotate
    bolometer_camera.transform = (
        rotate_z(raw_data["rotate"])
        * translate(*slit_centre)
        * rotate_basis(foil_normal, foil_basis_y)
        * translate(-centre_slit_local.x, -centre_slit_local.y, 0)
    )

    return bolometer_camera


def calcam_virtual_calibration(
    port: str = "6.5U", model_variant: str = "BC02"
) -> dict[str, float | tuple[int, int] | tuple[float, float, float]]:
    """generate virtual calibration parameters used in calcam for IRVB.
    This function calculate virtual pixel pitch, camera position, and camera orientation (target and roll)

    Parameters
    ----------
    port
        user-specified port name, by default "6.5U"
    model_variant
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
    # Load IRVB config. from .json file
    file = os.path.join(DATA_DIRECTORY, "IRVB.json")
    with open(file=file, mode="r") as f:
        raw_data = load(f)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][model_variant]
    except KeyError:
        raise KeyError(f"spcified parameters: {port} or {model_variant} are not defined.")

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

    # foil screen dimensions
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
    return (centre, basis_x, basis_y, width, height)


def _corners_coords(
    centre: Point3D, basis_x: Vector3D, basis_y: Vector3D, width: float, height: float
) -> list[Point3D]:
    """Calculate the rectangular corners"""
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
