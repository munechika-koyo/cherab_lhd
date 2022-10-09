"""
This module offers helper functions to load the LHD device.
"""
from __future__ import annotations
import os
from collections import defaultdict
from raysect.optical import rotate, World
from raysect.optical.library import RoughTungsten
from raysect.optical.material import AbsorbingSurface
from raysect.primitive.mesh import Mesh
from cherab.lhd.machine.material import RoughSUS316L


__all__ = ["load_pfc_mesh"]


def load_pfc_mesh(
    world: World,
    path: str = os.path.join(os.path.dirname(__file__), "geometry", "data", "RSMfiles"),
    reflections: bool = True,
    roughness: dict[str, float] = {"W": 0.29, "SUS": 0.0125},
) -> dict[str, Mesh]:
    """
    Loads LHD Plasma Facing Components mesh and connects it to
    Raysect :obj:`~raysect.core.scenegraph.world.World` instance.

    Parameters
    ----------
    world
        Raysect World instance
    path
        Path to directory containing .rsm files, by default `"../cherab/lhd/machine/geometry/data/RSMfiles"`
    reflections
        Reflection on/off, by default True
    roughness
        Roughness dict for PFC materials, by default `{"W": 0.29, "SUS": 0.0125}`.

    Returns
    -------
    dict[str, :obj:`~raysect.primitive.mesh.Mesh`]
        containing LHD device meshes

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.core import World
        >>> from cherab.lhd.machine import load_pfc_mesh
        >>>
        >>> world = World()
        >>> mesh = load_pfc_mesh(world, reflections=True)
    """
    # list o plasma facing components (= .rsm file name)
    pfc_list = ["vessel", "plates", "port_65u", "port_65l", "divertor"]

    # How many times each PFC element must be copy-pasted
    ncopy = defaultdict(lambda: 1)
    ncopy["plates"] = 5
    ncopy["divertor"] = 10

    if reflections:
        # set default roughness
        roughness.setdefault("W", 0.29)
        roughness.setdefault("SUS", 0.0125)

        materials = defaultdict(lambda: RoughSUS316L(roughness["SUS"]))
        materials["divertor"] = RoughTungsten(roughness["W"])
    else:
        materials = defaultdict(lambda: AbsorbingSurface())

    mesh = {}

    for pfc in pfc_list:
        try:
            mesh[pfc] = [
                Mesh.from_file(
                    os.path.join(path, f"{pfc}.rsm"),
                    parent=world,
                    material=materials[pfc],
                    name=pfc,
                )
            ]  # master element
            angle = 360.0 / ncopy[pfc]  # rotate around Z by this angle
            for i in range(1, ncopy[pfc]):  # copies of the master element
                mesh[pfc].append(
                    mesh[pfc][0].instance(
                        parent=world,
                        transform=rotate(0, 0, angle * i),
                        material=materials[pfc],
                        name=f"{pfc}-{i}",
                    )
                )
        except FileNotFoundError as e:
            print(e)
            continue
        except Exception as e:
            raise (e)

    return mesh


if __name__ == "__main__":
    # debag
    from raysect.core import print_scenegraph

    world = World()
    meshes = load_pfc_mesh(world, reflections=True)
    print_scenegraph(world)
