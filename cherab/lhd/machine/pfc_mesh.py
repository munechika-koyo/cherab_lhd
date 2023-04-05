"""This module offers helper functions to load the LHD device."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from raysect.optical import World, rotate
from raysect.optical.library import RoughTungsten
from raysect.optical.material import AbsorbingSurface
from raysect.primitive.mesh import Mesh

from cherab.lhd.machine.material import RoughSUS316L
from cherab.lhd.tools import Spinner

__all__ = ["load_pfc_mesh"]

# Path to directory containing .rsm files
DEFAULT_RSM_PATH = Path(__file__).parent.resolve() / "geometry" / "data" / "RSMfiles"

# List of Plasma Facing Components (same as filename "**.rsm")
PFC_LIST = ["vessel", "plates", "divertor"]  # ,"port_65u", "port_65l"]

# How many times each PFC element must be copy-pasted
NCOPY = defaultdict(lambda: 1)
NCOPY["plates"] = 5
NCOPY["divertor"] = 10
# NCOPY["port_65u"] = 10
# NCOPY["port_65l"] = 10

# Default roughness
ROUGHNESS = {"W": 0.29, "SUS": 0.0125}


def load_pfc_mesh(
    world: World,
    path: Path | str = DEFAULT_RSM_PATH,
    reflection: bool = True,
    roughness: dict[str, float] = ROUGHNESS,
) -> dict[str, list[Mesh]]:
    """Load LHD Plasma Facing Components mesh and connect it to Raysect
    :obj:`~raysect.core.scenegraph.world.World` instance.

    Parameters
    ----------
    world
        Raysect World instance
    path
        Path to directory containing .rsm files,
        by default ``"../cherab/lhd/machine/geometry/data/RSMfiles"``
    reflection
        Reflection on/off, by default True
    roughness
        Roughness dict for PFC materials, by default ``{"W": 0.29, "SUS": 0.0125}``.

    Returns
    -------
    dict[str, list[:obj:`~raysect.primitive.mesh.Mesh`]]
        containing LHD device meshes

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.core import World
        >>> from cherab.lhd.machine import load_pfc_mesh
        >>>
        >>> world = World()
        >>> mesh = load_pfc_mesh(world, reflection=True)
    """
    # validate path
    if isinstance(path, (str, Path)):
        path = Path(path)
    else:
        raise TypeError("2nd argument must be string or pathlib.Path instance.")

    if reflection:
        # set default roughness
        roughness.setdefault("W", 0.29)
        roughness.setdefault("SUS", 0.0125)

        materials = defaultdict(lambda: RoughSUS316L(roughness["SUS"]))
        materials["divertor"] = RoughTungsten(roughness["W"])
    else:
        materials = defaultdict(lambda: AbsorbingSurface())

    mesh = {}
    with Spinner(text="Loading PFCs...") as spinner:
        for pfc in PFC_LIST:
            try:
                mesh[pfc] = [
                    Mesh.from_file(
                        path / f"{pfc}.rsm",
                        parent=world,
                        material=materials[pfc],
                        name=pfc.capitalize(),
                    )
                ]  # master element
                angle = 360.0 / NCOPY[pfc]  # rotate around Z by this angle
                for i in range(1, NCOPY[pfc]):  # copies of the master element
                    mesh[pfc].append(
                        mesh[pfc][0].instance(
                            parent=world,
                            transform=rotate(0, 0, angle * i),
                            material=materials[pfc],
                            name=f"{pfc.capitalize()}-{i}",
                        )
                    )
                if value := getattr(materials[pfc], "roughness", None):
                    material_str = f"(roughness:{value:.5f})"
                else:
                    material_str = "(Absorbing Surface)"

                spinner.write(f"✅ {pfc.capitalize(): <9} {material_str} was loaded.")

            except FileNotFoundError as e:
                spinner.write(f"💥 {e}")
                continue
            except Exception as e:
                spinner.write(f"💥 {e}")

    return mesh


if __name__ == "__main__":
    # debag
    from raysect.core import print_scenegraph

    world = World()
    meshes = load_pfc_mesh(world, reflection=True)
    print_scenegraph(world)
