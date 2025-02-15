"""This module offers the helper function to easily set raytransfer material."""

from __future__ import annotations

from raysect.core.math import translate
from raysect.optical import World
from raysect.primitive import Cylinder

from cherab.lhd.tools.spinner import DummySpinner, Spinner

from ..indices import load_index_func
from .emitters import Discrete3DMeshRayTransferEmitter

__all__ = ["load_rte"]


# Constants
RMIN, RMAX = 2.0, 5.5  # [m]
ZMIN, ZMAX = -1.6, 1.6
ZONES = [
    "zone0",
    "zone1",
    "zone2",
    "zone3",
    "zone4",
    "zone11",
    "zone12",
    "zone13",
    "zone14",
    "zone15",
]


def load_rte(
    parent: World, zones: list[str] = ZONES, index_type: str = "cell", verbose: bool = True
) -> list[Cylinder]:
    """Helper function of loading RayTransfer Emitter using `.Discrete3DMeshRayTransferEmitter`.

    Parameters
    ----------
    parent : World
        Raysect world Node.
    zones : list[str]
        Zones of EMC3-EIRENE mesh.
    index_type : {"cell", "physics", "coarse"}, optional
        Type of indexing EMC3 grids, by default "cell".
        The index data must be created in advance using `.create_new_index`.
    verbose : bool, optional
        Whether to show progress bar, by default True.

    Returns
    -------
    list[`~raysect.primitive.cylinder.Cylinder`]
        List of primitives of cylinder.
    """

    emitters = []
    spinner = Spinner if verbose else DummySpinner
    base_text = "Loading RayTransfer Emitter "
    with spinner(text=base_text, timer=True) as sp:
        # Load index function
        for zone in zones:
            sp.text = base_text + f"({zone=}, {index_type=})"
            index_func, bins = load_index_func(zone=zone, index_type=index_type)

            # material as emitter
            material = Discrete3DMeshRayTransferEmitter(index_func, bins, integration_step=0.001)

            # primitive using cylinder
            shift = translate(0, 0, ZMIN)
            emitter = Cylinder(
                RMAX,
                ZMAX - ZMIN,
                transform=shift,
                parent=parent,
                material=material,
                name=f"RayTransferEmitter-{zone}",
            )

            emitters.append(emitter)

        sp.text = base_text
        sp.ok()

    return emitters
