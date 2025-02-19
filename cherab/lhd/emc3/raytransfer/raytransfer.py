"""This module offers the helper function to easily set raytransfer material."""

from __future__ import annotations

from raysect.core.math import translate
from raysect.optical import World
from raysect.primitive import Cylinder
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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
    parent: World,
    zones: list[str] = ZONES,
    integration_step: float = 1.0e-3,
    quiet: bool = False,
    **kwargs,
) -> tuple[Cylinder, int]:
    """Helper function of loading RayTransfer Emitter using `.Discrete3DMeshRayTransferEmitter`.

    Parameters
    ----------
    parent : `~raysect.optical.scenegraph.world.World`
        Raysect world scene-graph Node.
    zones : list[str]
        Zones of EMC3-EIRENE mesh.
    integration_step : float, optional
        Line integral step along the ray, by default 1.0 [mm].
    quiet : bool, optional
        Mute status messages, by default False.
    **kwargs
        Keyword arguments to pass to `.load_index_func`.

    Returns
    -------
    `~raysect.primitive.cylinder.Cylinder`
        Primitives of cylinder.
    int
        Number of voxels.

    Examples
    --------
    >>> from raysect.optical import World
    >>> world = World()
    >>> rte, bins = load_rte(world, zones=["zone0", "zone11"], index_type="coarse")
    """
    console = Console(quiet=quiet)
    progress = Progress(
        TimeElapsedColumn(),
        TextColumn("{task.description}"),
        SpinnerColumn("simpleDots"),
        console=console,
    )
    task_id = progress.add_task("Loading RayTransfer Emitter", total=1)

    with progress:
        # Load index function
        progress.update(task_id, description="Loading index function")
        index_func, dict_num_voxel = load_index_func(zones, quiet=True, **kwargs)
        bins = sum(dict_num_voxel.values())

        progress.update(task_id, description="Creating RayTransfer Emitter")

        # Create emitter material
        material = Discrete3DMeshRayTransferEmitter(
            index_func, bins, integration_step=integration_step
        )

        # Create Cylinder primitive
        shift = translate(0, 0, ZMIN)
        emitter = Cylinder(
            RMAX,
            ZMAX - ZMIN,
            transform=shift,
            parent=parent,
            material=material,
            name=f"RayTransferEmitter {zones}",
        )

        # Finalize progress
        progress.update(
            task_id,
            description=(
                "[bold green]RayTransfer Emitter loaded[/bold green] "
                f"(zones: {'+'.join(zones)}, bins: {bins})"
            ),
            advance=1,
        )
        progress.refresh()

    return emitter, bins
