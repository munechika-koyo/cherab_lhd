import os
from json import load
import numpy as np
from raysect.core.math import Point3D, Vector3D, translate, rotate_basis, rotate_vector, rotate_z
from raysect.primitive import Box, Subtract
from raysect.optical.material import AbsorbingSurface
from cherab.tools.observers import BolometerSlit
from .resistive import BolometerCamera, BolometerFoil


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")

XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)


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
    :obj:`.BolometerCamera`
        populated :obj:`.BolometerCamera` instance
    """
    # Load configuration from .json file
    file = os.path.join(DATA_DIRECTORY, "RB.json")
    with open(file=file, mode="r") as f:
        raw_data = load(f)
    
    return None
