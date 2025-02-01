"""Module offers helper functions related to resistive bolometers."""

from __future__ import annotations

import json

import numpy as np
from raysect.core import World
from raysect.core.math import Point3D, Vector3D
from raysect.core.scenegraph._nodebase import _NodeBase

from cherab.tools.observers import BolometerCamera, BolometerFoil, BolometerSlit

from ...tools.fetch import fetch_file

__all__ = ["load_resistive"]


def load_resistive(
    port: str = "6.5L", model_variant: str = "I", parent: _NodeBase | None = None
) -> BolometerCamera:
    """Helper function of generating a Resistive bolometer camera object.

    An bolometers' configuration is defined in json file, the name of which is ``"RB.json"``, which
    can be fetched from the remote repository.

    Parameters
    ----------
    port : {"6.5U", "6.5L", "3O", "8O"}, optional
        user-specified port name, by default ``"6.5L"``.
    model_variant : {"I", "O", "BC02-01", "BC02-04", ""}, optional
        The variant of bolometer model, by default ``"I"``.
    parent : `_NodeBase`, optional
        The parent node of this camera in the scenegraph, by default None.
        `~raysect.core.World` object is often used.

    Returns
    -------
    `~cherab.tools.observers.bolometry.BolometerCamera`
        Populated `~cherab.tools.observers.bolometry.BolometerCamera.BolometerCamera` instance.
    """
    # import configs as a resource
    with open(fetch_file("observer/RB.json"), "r") as file:
        raw_data = json.load(file)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][model_variant]
    except KeyError as err:
        raise KeyError(f"specified parameters: {port} or {model_variant} are not defined.") from err

    # ----------------------------------------------------------------------- #
    #  Build Bolometer Camera
    # ----------------------------------------------------------------------- #

    bolometer_camera = BolometerCamera(parent=parent, name=f"RB-{port}-{model_variant}")

    slit_centre, slit_basis_x, slit_basis_y, width, height = _centre_basis_and_dimensions(
        [Point3D(*slit) for slit in raw_data["slit"]]
    )
    slit = BolometerSlit(
        slit_id="slit",
        centre_point=slit_centre,
        basis_x=slit_basis_x,
        dx=width,
        basis_y=slit_basis_y,
        dy=height,
        parent=bolometer_camera,
        csg_aperture=True,
    )

    for id, foil_geom in raw_data["foil"].items():
        foil_geometry = _centre_basis_and_dimensions([Point3D(*foil_xyz) for foil_xyz in foil_geom])
        centre, basis_x, basis_y, width, height = foil_geometry

        # correct foil's direction (normal vector)
        if (centre.vector_to(slit.centre_point)).dot(basis_x.cross(basis_y)) < 0:
            basis_x = -basis_x

        foil = BolometerFoil(
            detector_id=f"Foil {id}",
            centre_point=centre,
            basis_x=basis_x,
            dx=width,
            basis_y=basis_y,
            dy=height,
            slit=slit,
            parent=bolometer_camera,
            units="Power",
            accumulate=False,
        )
        bolometer_camera.add_foil_detector(foil)

    # ----------------------------------------------------------------------- #
    #  Create CSG aperture
    # ----------------------------------------------------------------------- #
    # width = max(slit.dx, slit.dy) * 2.0
    # face = Box(Point3D(-width, -width, -slit.dz * 0.5), Point3D(width, width, slit.dz * 0.5))
    # slit_box = Box(
    #     lower=Point3D(-slit.dx * 0.5, -slit.dy * 0.5, -slit.dz * 0.6),
    #     upper=Point3D(slit.dx * 0.5, slit.dy * 0.5, slit.dz * 0.6),
    # )
    # _ = Subtract(
    #     face, slit_box, parent=slit, material=AbsorbingSurface(), name=f"{slit.name} - CSG Aperture"
    # )

    return bolometer_camera


def _centre_basis_and_dimensions(
    corners: list[Point3D],
) -> tuple[Point3D, Vector3D, Vector3D, float, float]:
    """Helper function of calculating centre, basis vectors, width and height of a rectangle from
    its corners.

    Parameters
    ----------
    corners : list[Point3D]
        A list of corners of a rectangle.

    Returns
    -------
    tuple[Point3D, Vector3D, Vector3D, float, float]
        Tuple of centre, basis vectors, width and height of a rectangle.
    """
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


if __name__ == "__main__":
    world = World()
    rb = load_resistive(parent=world)
    pass
