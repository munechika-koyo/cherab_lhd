"""Module to offer helper function to load plasma facing component meshes."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from plotly import graph_objects as go
from plotly.graph_objects import Figure
from raysect.optical import World, rotate_z
from raysect.optical.library import RoughTungsten
from raysect.optical.material.absorber import AbsorbingSurface

# from raysect.optical.material.lambert import Lambert
from raysect.optical.material.material import Material
from raysect.primitive.mesh import Mesh
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress
from rich.table import Table

from ..tools.fetch import fetch_file
from .material import RoughSUS316L

__all__ = ["load_pfc_mesh", "visualize_pfc_meshes"]

# TODO: omtimization of roughness
ROUGHNESS_SUS = 0.0125
ROUGHNESS_W = 0.29

# List of Plasma Facing Components (filename is "**.rsm")
COMPONENTS: dict[str, tuple[str, Material, float | None]] = {
    # name: (filename, material class, roughness)
    "Vacuum Vessel": ("vessel", RoughSUS316L, ROUGHNESS_SUS),
    "First Wall": ("plates", RoughSUS316L, ROUGHNESS_SUS),
    "Divertor": ("divertor", RoughTungsten, ROUGHNESS_W),
    # "Port 65 Upper": ("port_65u", RoughSUS316L, ROUGHNESS_SUS),
    # "Port 65 Lower": ("port_65l", RoughSUS316L, ROUGHNESS_SUS),
}

# How many times each PFC element must be copy-pasted
NCOPY: dict[str, int] = defaultdict(lambda: 1)
NCOPY["First Wall"] = 5
NCOPY["Divertor"] = 10
# NCOPY["Port 65 Upper"] = 10
# NCOPY["Port 65 Lower"] = 10

# Offset toroidal angle
ANG_OFFSET: dict[str, float] = defaultdict(lambda: 0.0)


def load_pfc_mesh(
    world: World,
    custom_components: dict[str, tuple[str, Material, float | None]] | None = None,
    reflection: bool = True,
    quiet: bool = False,
    **kwargs,
) -> dict[str, list[Mesh]]:
    """Load plasma facing component meshes.

    Each mesh allows the user to use an user-defined material which inherites
    `~raysect.optical.material.material.Material`.

    Parameters
    ----------
    world : `~raysect.optical.world.World`
        The world scenegraph to which the meshes will be added.
    custom_components : dict[str, tuple[str, Material, float | None]], optional
        Custom components to load, by default None.
        The structure of the dictionary is as follows:
        ``{"Component Name": ("path", Material class, roughness)}``.
        If the custom component is given, it is merged with the default components.
    reflection : bool, optional
        Whether or not to consider reflection light, by default True.
        If ``False``, all of meshes' material are replaced to
        `~raysect.optical.material.absorber.AbsorbingSurface`.
    quiet : bool, optional
        If ``True``, it suppresses the progress table, by default False.
    **kwargs
        Keyword arguments to pass to `.fetch_file`.

    Returns
    -------
    dict[str, list[`~raysect.primitive.mesh.mesh.Mesh`]]
        Containing mesh name and `~raysect.primitive.mesh.mesh.Mesh` objects.

    Examples
    --------
    .. code-block:: python

        from raysect.optical import World

        world = World()
        meshes = load_pfc_mesh(world, reflection=True)
    """
    # Merge user-defined components with default components
    if custom_components is not None:
        components = COMPONENTS | custom_components
    else:
        components = COMPONENTS

    # Fetch meshes in advance
    paths_to_rsm = {}
    for mesh_name, (filename, _, _) in components.items():
        path = fetch_file(f"machine/{filename}.rsm", **kwargs)
        paths_to_rsm[mesh_name] = path

    # Create progress bar and add task
    progress = Progress(transient=True)
    task_id = progress.add_task("", total=len(components))

    if not quiet:
        # Create Table of the status of loading
        table = Table(title="Plasma Facing Components", show_footer=False)
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Path to file", justify="left", style="magenta")
        table.add_column("Material", justify="center", style="green")
        table.add_column("Roughness", justify="center", style="yellow")
        table.add_column("Loaded", justify="center")

        # Create Group to show progress bar and table
        progress_group = Group(table, progress)
    else:
        progress_group = Group(progress)

    # Load meshes
    meshes = {}
    with Live(progress_group, auto_refresh=False, console=Console(quiet=quiet)) as live:
        for mesh_name, (_, material_cls, roughness) in components.items():
            progress.update(task_id, description=f"Loading {mesh_name}...")
            live.refresh()
            try:
                # Configure material
                if not reflection:
                    material_cls = AbsorbingSurface
                    roughness = None

                if roughness is not None:
                    material = material_cls(roughness=roughness)
                else:
                    material = material_cls()

                # ================================
                # Load mesh
                # ================================
                # Load master element
                meshes[mesh_name] = [
                    Mesh.from_file(
                        paths_to_rsm[mesh_name],
                        parent=world,
                        transform=rotate_z(ANG_OFFSET[mesh_name]),
                        material=material,
                        name=f"{mesh_name} 1" if NCOPY[mesh_name] > 1 else f"{mesh_name}",
                    )
                ]
                # Copy of the master element
                angle = 360.0 / NCOPY[mesh_name]
                for i in range(1, NCOPY[mesh_name]):
                    meshes[mesh_name].append(
                        meshes[mesh_name][0].instance(
                            parent=world,
                            transform=rotate_z(angle * i + ANG_OFFSET[mesh_name]),
                            material=material,
                            name=f"{mesh_name} {i + 1}",
                        )
                    )

                # Save the status of loading
                _status = "✅"
            except Exception as e:
                _status = f"❌ ({e})"
            finally:
                if not quiet:
                    table.add_row(
                        mesh_name,
                        paths_to_rsm[mesh_name],
                        material_cls.__name__,
                        str(roughness),
                        _status,
                    )
                progress.advance(task_id)

        progress.update(task_id, visible=False)
        live.refresh()

    return meshes


def visualize_pfc_meshes(
    meshes: dict[str, list[Mesh]], fig: Figure | None = None, fig_size: tuple[int, int] = (700, 500)
) -> Figure:
    """Visualize plasma facing component meshes.

    This function allows the user to visualize the plasma facing component meshes using plotly.

    Parameters
    ----------
    meshes : dict[str, list[`~raysect.primitive.mesh.mesh.Mesh`]]
        Containing mesh name and `~raysect.primitive.mesh.mesh.Mesh` objects.
    fig : `~plotly.graph_objects.Figure`, optional
        Plotly Figure object, by default `~plotly.graph_objects.Figure`.
    fig_size : tuple[int, int], optional
        Figure size, by default (700, 500) pixel.

    Returns
    -------
    `~plotly.graph_objects.Figure`
        Plotly Figure object.

    Examples
    --------
    .. code-block:: python

        from raysect.optical import World
        from cherab.lhd.machine.pfc_mesh import load_pfc_mesh, visualize_pfc_meshes

        world = World()
        meshes = load_pfc_mesh(world)

        # Drop some meshes
        meshes.pop("Vacuum Vessel")
        meshes.pop("Divertor")

        fig = visualize_pfc_meshes(meshes, fig_size=(700, 500))
        fig.show()

    The above codes automatically launch a browser to show the figure when it is executed in
    the python interpreter like the following picture:

    .. image:: ../_static/images/visualize_pfc_meshes.webp
    """
    if fig is None or not isinstance(fig, Figure):
        fig = go.Figure()

    for _, mesh_list in meshes.items():
        for mesh in mesh_list:
            # Rotate mesh by its transform matrix
            transform = mesh.to_root()
            rotation = np.array([[transform[i, j] for j in range(3)] for i in range(3)])
            x, y, z = rotation @ mesh.data.vertices.T
            i, j, k = mesh.data.triangles.T

            # Create Mesh3d object
            mesh3D = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                flatshading=True,
                colorscale=[[0, "#e5dee5"], [1, "#e5dee5"]],
                intensity=z,
                name=f"{mesh.name}",
                text=f"{mesh.name}",
                showscale=False,
                showlegend=True,
                lighting=dict(
                    ambient=0.18,
                    diffuse=1,
                    fresnel=0.1,
                    specular=1,
                    roughness=0.1,
                    facenormalsepsilon=0,
                ),
                lightposition=dict(x=3000, y=3000, z=10000),
                hovertemplate=f"<b>{mesh.name}</b><br>" + "x: %{x}<br>y: %{y}<br>z: %{z}<br>"
                "<extra></extra>",
            )

            fig.add_trace(mesh3D)

    fig.update_layout(
        template="plotly_dark",
        title_text="Device",
        title_x=0.5,
        width=fig_size[0],
        height=fig_size[1],
        scene_aspectmode="data",
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.0, y=1.0, z=0.8),
        ),
        margin=dict(r=10, l=10, b=10, t=35),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_xaxis_visible=False,
        scene_yaxis_visible=False,
        scene_zaxis_visible=False,
    )

    return fig
