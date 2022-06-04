import os
from json import load
from raysect.core import World
from raysect.core.math import Point3D
from raysect.primitive import Box, Subtract
from raysect.optical.material import AbsorbingSurface
from cherab.tools.observers import BolometerCamera, BolometerSlit, BolometerFoil
from .load_irvb import _centre_basis_and_dimensions


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")


def load_resistive(port="6.5L", model_variant="I", parent=None):
    """helper function of generating a Resistive bolometer camera object.
    An bolometers' configuration is defined in json file, the name of which is "RB.json" stored in ../data folder.

    Parameters
    ----------
    port : str
        user-specified port name, by default "6.5L"
    model_variant : str
        The variant of bolometer model, by default "I"
    parent : :obj:`~raysect.core.scenegraph.node.Node`, optional
        The parent node of this camera in the scenegraph, often
        an optical :obj:`~raysect.core.scenegraph.world.World` object, by default None

    Returns
    -------
    :obj:`~cherab.tools.observers.bolometry.BolometerCamera`
        populated :obj:`~cherab.tools.observers.bolometry.BolometerCamera.BolometerCamera` instance
    """
    # Load configuration from .json file
    file = os.path.join(DATA_DIRECTORY, "RB.json")
    with open(file=file, mode="r") as f:
        raw_data = load(f)

    # extract user-specified IRVB model
    try:
        raw_data = raw_data[port][model_variant]
    except KeyError:
        raise KeyError(f"spcified parameters: {port} or {model_variant} are not defined.")

    # ----------------------------------------------------------------------- #
    #  Build Bolometer Camera
    # ----------------------------------------------------------------------- #

    bolometer_camera = BolometerCamera(parent=parent, name=f"LHD-RB-{port}-{model_variant}")

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
    )

    for id, foil_geom in raw_data["foil"].items():
        foil_geometry = _centre_basis_and_dimensions([Point3D(*foil_xyz) for foil_xyz in foil_geom])
        centre, basis_x, basis_y, width, height = foil_geometry
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
    width = max(slit.dx, slit.dy) * 2.0
    face = Box(Point3D(-width, -width, -slit.dz * 0.5), Point3D(width, width, slit.dz * 0.5))
    slit_box = Box(
        lower=Point3D(-slit.dx * 0.5, -slit.dy * 0.5, -slit.dz * 0.6),
        upper=Point3D(slit.dx * 0.5, slit.dy * 0.5, slit.dz * 0.6),
    )
    _ = Subtract(face, slit_box, parent=slit, material=AbsorbingSurface(), name=f"{slit.name} - CSG Aperture")

    return bolometer_camera


if __name__ == "__main__":
    world = World()
    rb = load_resistive(parent=world)
    pass
