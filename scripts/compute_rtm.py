"""Calculate Ray-Transfer Matrix of bolometer cameras.

This script is to compute Ray-Transfer Matrix (RTM) with resistive bolometers and IRVBs.
Calculated RTM is saved as a `*.nc` file in the `rtm` directory.
"""

# %%
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from raysect.optical import World
from raysect.optical.observer import FullFrameSampler2D  # type: ignore
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from cherab.lhd.emc3.raytransfer import load_rte
from cherab.lhd.machine import load_pfc_mesh
from cherab.lhd.observer.bolometer import load_irvb, load_resistive
from cherab.tools.raytransfer import RayTransferPipeline0D, RayTransferPipeline2D  # type: ignore

BASE = Path(__file__).parent

# Get arguments
# -------------
parser = argparse.ArgumentParser(description="Compute RTM for bolometer cameras")
parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
args = parser.parse_args()
quiet = args.quiet

# Set console
console = Console(quiet=quiet)

# %%
# Set tasks
# ---------
rb_tasks = [("6.5L", "I"), ("6.5L", "O"), ("8O", "")]
irvb_tasks = [("6.5U", "CC01_04"), ("6.5L", "BC02")]
tasks = rb_tasks + irvb_tasks

# %%
# Create scene-graph
# ------------------
# Here we add objects (machine, emitters) to the scene root (World).

# scene world
world = World()

# machine mesh
mesh = load_pfc_mesh(world, reflection=False, quiet=quiet)

# Ray Transfer Emitter (zone0 + zone11)
index_type = "coarse"
rte, bins = load_rte(
    world,
    zones=["zone0", "zone11"],
    index_type=index_type,
    integration_step=1.0e-3,
    quiet=quiet,
)

# %%
# Create Live progress
# --------------------
# Group of progress bars;
# Some are always visible, others will disappear when progress is complete
bolo_progress = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
    console=console,
)
bolo_current_progress = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("[bold purple]{task.fields[action]}"),
    SpinnerColumn("simpleDots"),
    console=console,
)
foil_progress = Progress(
    TextColumn("[bold blue]Progress for foils {task.percentage:.0f}%"),
    BarColumn(),
    TextColumn("({task.completed} of {task.total} foil done)"),
    console=console,
)
# overall progress bar
overall_progress = Progress(
    TimeElapsedColumn(),
    BarColumn(),
    TextColumn("{task.description}"),
    console=console,
)
# Set progress panel
progress_panel = Group(
    Panel(
        Group(bolo_progress, bolo_current_progress, foil_progress),
        title="RTM for Bolometers",
        title_align="left",
    ),
    overall_progress,
)

# Set tasks
overall_task_id = overall_progress.add_task("", total=len(tasks))


# %%
# Compute RTM
# -----------
# Here we compute RTM for each bolometer camera.
# Each RTM is stored as an `xarray.Dataset` in a separate group named after the bolometer name.
# Then, the dataset is saved as a `*.nc` file in the `rtm` directory.
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_path = BASE / "rtm" / f"{time_now}.nc"

with Live(progress_panel):
    i_task = 0
    # --------------------
    # Resistive Bolometers
    # --------------------
    # Calculate RTM for resistive bolometers
    for port, model in rb_tasks:
        # Set overall task
        overall_progress.update(
            overall_task_id,
            description=f"[bold #AAAAAA]({i_task} out of {len(tasks)} bolometers done)",
        )
        # Initialize current task
        current_task_id = bolo_current_progress.add_task("", action="Preprocessing")

        # List for storing RTM of each foil
        rtm = []

        # load bolometer camera
        bolo = load_resistive(port=port, model_variant=model, parent=world)

        # Set task for each progress
        bolo_task_id = bolo_progress.add_task(f"{bolo.name}")
        foil_task_id = foil_progress.add_task("", total=len(bolo))
        bolo_current_progress.update(current_task_id, action="Rendering")
        for foil in bolo:
            rtp = RayTransferPipeline0D(name=foil.name)
            foil.pipelines = [rtp]
            foil.pixel_samples = 5e4
            foil.min_wavelength = 600
            foil.max_wavelength = 601
            foil.spectral_rays = 1
            foil.spectral_bins = bins
            foil.observe()

            # Store values
            rtm += [rtp.matrix]

            # update progress
            foil_progress.update(foil_task_id, advance=1, refresh=True)

        foil_progress.update(foil_task_id, visible=False)

        bolo_current_progress.update(current_task_id, action="Saving")
        # Set config info
        config = {
            "bolometer name": bolo.name,
            "bolometer Number of foils": len(bolo),
            "foil spectral bins": bolo[0].spectral_bins,
            "foil pixel samples": bolo[0].pixel_samples,
            "foil wavelength range": (bolo[0].min_wavelength, bolo[0].max_wavelength),
        }
        xr.Dataset(
            data_vars=dict(
                rtm=(
                    ["foil", "voxel"],
                    np.asarray_chkfinite(rtm) / (4.0 * np.pi),  # [m^3 sr] -> [m^3]
                    dict(units="m^3", long_name="ray transfer matrix"),
                ),
                area=(
                    ["foil"],
                    np.asarray([foil.collection_area for foil in bolo]),
                    dict(units="m^2", long_name="foil area"),
                ),
            ),
            coords=dict(
                foil=(
                    ["foil"],
                    np.arange(1, len(bolo) + 1, dtype=int),
                    dict(units="ch", long_name="foil channel"),
                ),
                voxel=(
                    ["voxel"],
                    np.arange(bins, dtype=int),
                    dict(long_name="voxel index"),
                ),
            ),
            attrs=config,
        ).to_netcdf(save_path, mode="a", group=f"{bolo.name}")

        # update progress
        bolo_current_progress.stop_task(current_task_id)
        bolo_current_progress.update(current_task_id, visible=False)

        bolo_progress.stop_task(bolo_task_id)
        bolo_progress.update(bolo_task_id, description=f"[bold green]{bolo.name} done!")

        overall_progress.update(overall_task_id, advance=1, refresh=True)

        i_task += 1

    # --------------------------
    # Infra-Red Video Bolometers
    # --------------------------
    # Calculate RTM for IRVBs
    for port, flange in irvb_tasks:
        # Update overall task
        overall_progress.update(
            overall_task_id,
            description=f"[bold #AAAAAA]({i_task} out of {len(tasks)} bolometers done)",
        )

        # Initialize current task
        current_task_id = bolo_current_progress.add_task("", action="Preprocessing")

        # Load IRVB
        bolo = load_irvb(port=port, flange=flange, parent=world)

        rtp = RayTransferPipeline2D(name=f"{bolo.name}")
        pipelines = [rtp]
        sampler = FullFrameSampler2D()
        foil_detector = bolo.foil_detector
        foil_detector.frame_sampler = sampler
        foil_detector.pipelines = pipelines
        foil_detector.pixel_samples = 5e3
        foil_detector.min_wavelength = 600
        foil_detector.max_wavelength = 601
        foil_detector.spectral_rays = 1
        foil_detector.spectral_bins = bins
        foil_detector.quiet = True

        # Set task for each progress
        bolo_task_id = bolo_progress.add_task(f"{bolo.name}")
        bolo_current_progress.update(current_task_id, action="Rendering")

        # Rendering
        bolo.observe()

        bolo_current_progress.update(current_task_id, action="Saving")

        # save config info as text
        config = {
            "bolometer name": bolo.name,
            "foil spectral bins": foil_detector.spectral_bins,
            "foil pixels": foil_detector.pixels,
            "foil pixel samples": foil_detector.pixel_samples,
            "foil wavelength range": (foil_detector.min_wavelength, foil_detector.max_wavelength),
            "units": "m^3",
            "pixel_area": (foil_detector.width / foil_detector.pixels[0]) ** 2,
        }

        # Save RTM as a xarray.DataAarray
        nx, ny = foil_detector.pixels
        dx = 1e2 * foil_detector.width / nx  # [m] -> [cm]
        da = xr.DataArray(
            data=np.asarray_chkfinite(rtp.matrix) / (4.0 * np.pi),  # [m^3 sr] -> [m^3]
            dims=["x", "y", "voxel"],
            coords=dict(
                x=(
                    ["x"],
                    np.linspace(dx * 0.5, dx * (nx - 0.5), nx, endpoint=True),
                    dict(units="cm", long_name="width"),
                ),
                y=(
                    ["y"],
                    np.linspace(dx * 0.5, dx * (ny - 0.5), ny, endpoint=True),
                    dict(units="cm", long_name="height"),
                ),
                voxel=(
                    ["voxel"],
                    np.arange(bins, dtype=int),
                    dict(long_name="voxel index"),
                ),
            ),
            attrs=config,
            name="rtm",
        )
        da.to_netcdf(save_path, mode="a", group=f"{bolo.name}")

        # update progress
        bolo_current_progress.stop_task(current_task_id)
        bolo_current_progress.update(current_task_id, visible=False)

        bolo_progress.stop_task(bolo_task_id)
        bolo_progress.update(bolo_task_id, description=f"[bold green]{bolo.name} done!")

        overall_progress.update(overall_task_id, advance=1, refresh=True)

        i_task += 1

    # Finalize progress
    overall_progress.update(
        overall_task_id,
        description=f"[bold green]{len(tasks)} bolometers done!",
    )

# %%
# Post process
# ------------
# Save config info for the whole
config_total = {
    "primitives material": world.primitives[1].material.__class__.__name__,
    "machine PFCs": str(mesh),
    "cell index type": index_type,
    "zones": "zone0+zone11",
    "RTM bins": bins,
}
xr.Dataset(attrs=config_total).to_netcdf(save_path, mode="a")
console.print(f"[bold green]RTM data saved at[/bold green] {save_path}")
