"""Module offers helper functions related to a IRVB camera."""
from __future__ import annotations

import json
from importlib.resources import files

from raysect.core.math import Point3D, Vector3D, rotate_basis, rotate_z, translate
from raysect.optical.material import AbsorbingSurface
from raysect.optical.observer import TargettedCCDArray
from raysect.primitive import Box, Subtract

from cherab.tools.observers import BolometerSlit

from ..imaging.pinhole import PinholeCamera
from .irvb import IRVBCamera

__all__ = ["load_irvb", "load_irvb_as_pinhole_camera"]

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
    with files("cherab.lhd.observer.bolometer.data").joinpath("IRVB.json").open("r") as file:
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
    else:
        raise KeyError(f"IRVB model (port:{port}, flange:{flange}) is obsolete.")

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
    with files("cherab.lhd.observer.bolometer.data").joinpath("IRVB.json").open("r") as file:
        raw_data = json.load(file)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][flange]
    except KeyError as err:
        raise KeyError(f"spcified parameters: {port} or {flange} are not defined.") from err

    # Construct Foil and Slit from port local coordinates
    if raw_data.get("basis_x"):
        FOIL_WIDTH, FOIL_HEIGHT = raw_data["foil_size"]
        foil_centre = Point3D(*raw_data["foil_centre"])
        slit_centre_in_local = Point3D(*raw_data["slit_centre_in_local"])
        basis_x = Vector3D(*raw_data["basis_x"]).normalise()
        basis_y = Vector3D(*raw_data["basis_y"]).normalise()
        PIXELS = raw_data["pixels"]
        foil_forward = basis_x.cross(basis_y)
    else:
        raise KeyError(f"IRVB model (port:{port}, flange:{flange}) is obsolete.")

    # ----------------------------------- #
    # Construct Pinhole Camera object
    # ----------------------------------- #

    bolometer_camera = PinholeCamera(
        pixels=PIXELS,
        width=FOIL_WIDTH,
        focal_length=slit_centre_in_local.z,
        pinhole_point=(slit_centre_in_local.x, slit_centre_in_local.y),
        parent=parent,
        name=f"IRVB-{port}-{flange}",
    )

    # camera rotate
    bolometer_camera.transform = (
        rotate_z(raw_data["rotate"]) * translate(*foil_centre) * rotate_basis(foil_forward, basis_y)
    )

    return bolometer_camera


if __name__ == "__main__":
    from raysect.optical import World

    world = World()
    irvb = load_irvb(port="6.5L", parent=world)

    print("debug")
