import os
from collections import defaultdict
from raysect.optical import rotate
from raysect.optical.library import RoughBeryllium, RoughIron, RoughTungsten
from raysect.optical.material import AbsorbingSurface
from raysect.primitive.mesh import Mesh
from cherab.phix.machine.material import RoughSUS316L


def load_pfc_mesh(
    world,
    path=os.path.join(os.path.dirname(__file__), "geometry", "data", "RSMfiles"),
    reflections=True,
    roughness={"Be": 0.26, "W": 0.29, "Fe": 0.13, "SUS": 0.0125},
):
    """
    Loads LHD Plasma Facing Components mesh and connects it to
    Raysect :obj:`~raysect.core.scenegraph.world.World` instance.

    Parameters
    ----------
    world : :obj:`~raysect.core.scenegraph.world.World`
        Raysect World instance
    path : str, optional
        Path to directory containing .rsm files, by default "../cherab/lhd/machine/geometry/data/RSMfiles"
    reflections : bool, optional
        Reflection on/off, by default True
    roughness : dict, optional
        Roughness dict for PFC materials, by default {"Be": 0.26, "W": 0.29, "Fe": 0.13, "SUS": 0.0125}.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.core import World
        >>> from cherab.lhd.machine import load_pfc_mesh
        >>>
        >>> world = World()
        >>> mesh = load_pfc_mesh(world, reflections=True)
    """

    pfc_list = ["vessel", "plates", "port_65u", "port_65l", "divertor"]

    # How many times each PFC element must be copy-pasted
    ncopy = defaultdict(lambda: 1)
    ncopy["plates"] = 5
    ncopy["divertor"] = 10

    if reflections:
        # set default roughness
        roughness.setdefault("Be", 0.26)
        roughness.setdefault("W", 0.29)
        roughness.setdefault("Fe", 0.13)
        roughness.setdefault("SUS", 0.0125)

        materials = defaultdict(lambda: RoughSUS316L(roughness["SUS"]))
        materials["divertor"] = RoughTungsten(roughness["W"])
    else:
        materials = defaultdict(lambda: AbsorbingSurface())

    mesh = {}

    for pfc in pfc_list:
        mesh[pfc] = [
            Mesh.from_file(os.path.join(path, f"{pfc}.rsm"), parent=world, material=materials[pfc])
        ]  # master element
        angle = 360.0 / ncopy[pfc]  # rotate around Z by this angle
        for i in range(1, ncopy[pfc]):  # copies of the master element
            mesh[pfc].append(
                mesh[pfc][0].instance(parent=world, transform=rotate(0, 0, angle * i), material=materials[pfc])
            )

    return mesh


if __name__ == "__main__":
    pass
