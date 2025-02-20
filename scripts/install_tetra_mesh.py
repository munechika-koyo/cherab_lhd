"""Script to generate tetrahedral mesh from given zones."""

from __future__ import annotations

import json
from importlib.resources import as_file, files

import numpy as np
from pooch import file_hash
from raysect.primitive.mesh import TetraMeshData
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from cherab.lhd.emc3 import Grid
from cherab.lhd.emc3.cython import tetrahedralize  # type: ignore
from cherab.lhd.tools.fetch import PATH_TO_STORAGE, get_registries


def worker(zones: tuple[str, ...], console: Console | None = None) -> tuple[str, str]:
    """Generate tetrahedral mesh from given zones.

    If specifying one zone, the tetrahedral mesh will be saved like ``zone0.rsm``.
    If specifying multiple zones, the tetrahedral mesh will be saved like ``zone0+zone1.rsm``.

    Parameters
    ----------
    zones : tuple[str,...]
        Zones to generate tetra mesh from.
        If you want to pass one zone, use like ``("zone0",)``.
        Otherwise, use like ``("zone0", "zone1", ...)``.
    console : Console, optional
        Rich console object to log messages.
        If not provided, a new console object will be created.

    Returns
    -------
    str
        Name of the tetra mesh file.
    str
        Sha256 Hash of the tetra mesh file.
    """
    if console is None:
        console = Console()

    # Generate vertices and cells
    grids = [Grid(zone) for zone in zones]
    verts = np.vstack([grid.generate_vertices() for grid in grids])
    cells = []

    start_index = 0
    for grid in grids:
        cell = grid.generate_cell_indices()
        cells.append(cell + start_index)
        start_index += cell.max() + 1
    cells = np.vstack(cells)

    console.log("Vertices and cells generated!")

    # Tetrahedralize
    tetrahedral = tetrahedralize(cells)
    console.log("Tetrahedralization done!")

    # Make tetra mesh
    tetra = TetraMeshData(verts, tetrahedral, tolerant=False)
    console.log("Tetra mesh created!")

    # Save tetra mesh
    tetra_mesh = "+".join(zones)
    tetra_path = PATH_TO_STORAGE / "tetra" / f"{tetra_mesh}.rsm"
    tetra.save(tetra_path)
    console.log(f"Tetra mesh saved at {tetra_path}")

    return tetra_path.name, file_hash(tetra_path)


def update_registry(registry_key: str, hash: str) -> None:
    """Update registries.json with new hash.

    Parameters
    ----------
    registry_key : str
        Key to update in the registries.json.
    hash : str
        Hash to update in the registries.json.
    """
    registry = get_registries()
    registry.update({registry_key: hash})
    with as_file(files("cherab.lhd.tools").joinpath("registries.json")) as file:
        with file.open("w") as f:
            json.dump(registry, f, indent=2)


if __name__ == "__main__":
    console = Console()

    # %%
    # Define tasks
    # ============
    # Tasks are defined as a list of tuples.
    # Each tuple contains the zones to generate tetra mesh from.
    # If you want to pass one zone, use like ("zone0",).
    # Otherwise, use like ("zone0", "zone1", ...).
    tasks = [
        # ("zone0",),  # Single zone
        ("zone0", "zone11"),  # Multiple zones combined to one mesh
    ]

    with Progress(
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn("simpleDots"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("", total=len(tasks))
        for zones in tasks:
            # Generate tetra mesh
            progress.update(task_id, description=f"Generating tetra mesh for {'+'.join(zones)}")
            key, hash = worker(zones, console)

            # Update registry with new hash
            update_registry(key, hash)
            progress.advance(task_id)

        progress.update(task_id, description="[green bold]All tetra meshes generated!")
        progress.refresh()
