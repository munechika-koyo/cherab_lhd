"""Module to convert STL files to RSM files.

STL means Standard Tessellation Language, which is a file format native to the stereolithography CAD
software created by 3D Systems.
RSM means Raysect Mesh files containing a K-D tree structure as well as basic mesh information.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from pathlib import Path

from raysect.optical import World
from raysect.primitive import import_stl
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ...tools.fetch import PATH_TO_STORAGE


def stl_to_rsm(stl_dir: Path | str, scale: float = 1.0, update=False) -> None:
    """Convert all STL files in a directory to RSM files.

    The conversion process is performed in parallel using a thread pool.

    Parameters
    ----------
    stl_dir : Path | str
        Path to the directory containing the STL files.
    scale : float, optional
        Scaling factor to apply to the STL files, by default 1.0.
    update : bool, optional
        If True, it forces to update the RSM files even if they already exist, by default False.
    """

    def worker(task_id, pfc_path, progress):
        progress.start_task(task_id)
        world = World()
        mesh = import_stl(pfc_path, scaling=scale, parent=world)
        mesh.save(path_to_machine_storage / pfc_path.with_suffix(".rsm").name)
        progress.update(task_id, advance=1)

    # Create storage directory
    path_to_machine_storage = PATH_TO_STORAGE / "machine"
    path_to_machine_storage.mkdir(parents=True, exist_ok=True)

    # Create tasks (get SLT file paths)
    tasks = []
    for pfc_path in Path(stl_dir).glob("*.stl"):
        if update or not (path_to_machine_storage / pfc_path.with_suffix(".rsm").name).exists():
            tasks.append(pfc_path)

    if not tasks:
        print("No STL files to convert.")
        return

    # Create progress bar
    progress = Progress(
        SpinnerColumn(finished_text=":white_check_mark:"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    )

    # Run tasks in parallel
    num_pool = min(cpu_count(), len(tasks))
    with progress:
        with ThreadPoolExecutor(num_pool) as executor:
            for task in tasks:
                task_id = progress.add_task(
                    f"[cyan]Converting {pfc_path.name}", total=1, start=False
                )
                executor.submit(worker, task_id, task, progress)
