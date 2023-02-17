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


DEFAULT_RSM_PATH = Path(__file__).parent.resolve() / "geometry" / "data" / "RSMfiles"


def load_pfc_mesh(
    world: World,
    path: Path | str = DEFAULT_RSM_PATH,
    reflections: bool = True,
    roughness: dict[str, float] = {"W": 0.29, "SUS": 0.0125},
) -> dict[str, list[Mesh]]:
    """Loads LHD Plasma Facing Components mesh and connects it to Raysect
    :obj:`~raysect.core.scenegraph.world.World` instance.

    Parameters
    ----------
    world
        Raysect World instance
    path
        Path to directory containing .rsm files,
        by default ``"../cherab/lhd/machine/geometry/data/RSMfiles"``
    reflections
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
        >>> mesh = load_pfc_mesh(world, reflections=True)
    """
    # validate path
    if isinstance(path, (str, Path)):
        path = Path(path)
    else:
        raise TypeError("2nd argument must be string or pathlib.Path instance.")

    # list o plasma facing components (= .rsm file name)
    pfc_list = ["vessel", "plates", "divertor"]  # , "port_65u", "port_65l"]

    # How many times each PFC element must be copy-pasted
    ncopy = defaultdict(lambda: 1)
    ncopy["plates"] = 5
    ncopy["divertor"] = 10
    # ncopy["port_65u"] = 10
    # ncopy["port_65l"] = 10

    if reflections:
        # set default roughness
        roughness.setdefault("W", 0.29)
        roughness.setdefault("SUS", 0.0125)

        materials = defaultdict(lambda: RoughSUS316L(roughness["SUS"]))
        materials["divertor"] = RoughTungsten(roughness["W"])
    else:
        materials = defaultdict(lambda: AbsorbingSurface())

    mesh = {}
    with Spinner(text="Loading PFCs...") as spinner:
        for pfc in pfc_list:
            try:
                mesh[pfc] = [
                    Mesh.from_file(
                        path / f"{pfc}.rsm",
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
                if hasattr(materials[pfc], "roughness"):
                    material_str = f"(roughness:{getattr(materials[pfc], 'roughness'):.5f})"
                else:
                    material_str = "(Absorbing Surface)"

                spinner.write(f"✅ {pfc: <9} {material_str} was loaded.")

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
    meshes = load_pfc_mesh(world, reflections=True)
    print_scenegraph(world)
