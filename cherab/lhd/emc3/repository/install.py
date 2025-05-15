"""Module to provide useful functions around Installing EMC3-related data."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Literal

import h5py  # noqa: F401
import numpy as np
import xarray as xr
from rich.progress import Progress

from ...tools.fetch import PATH_TO_STORAGE
from ..cython.utility import compute_centers
from .parse import DataParser
from .utility import exist_path_validate, path_validate

__all__ = ["install_grids", "install_indices", "install_center_points", "install_data"]

# Define zone labels
ZONES = [f"zone{i}" for i in range(0, 21 + 1)]


def install_grids(
    path: Path | str,
    mode: Literal["w", "a"] = "a",
    save_dir: Path | str = PATH_TO_STORAGE / "emc3",
) -> None:
    """Install EMC3-EIRENE grid into netCDF file.

    EMC3-EIRENE grid data is stored in a text file originally.
    This function parses the text file and save the grid data into a netCDF file.

    The value of :math:`R_\\mathrm{ax}` coordinates of the magnetic axis is parsed from a filename
    (e.g. ``grid-360.text`` means $R = 3.6$ m). :math:`Z_\\mathrm{ax}` is always regarded as 0.0.
    The name of the saved file is the same as the name of the file to be loaded
    (e.g. ``grid-360.nc``).

    Parameters
    ----------
    path : Path | str
        Path to the original text file written about grid coordinates at each zone.
    mode : {"w", "a"}, optional
        Mode to open the netCDF file, by default "a".
        - "w": write mode (overwrite if exists)
        - "a": append mode (create if not exists)
    save_dir : Path | str, optional
        Directory path to save the netCDF file, by default ``cherab/lhd/emc3/`` under the user's
        cache directory.
    """
    # Validate paths
    path = exist_path_validate(path)

    # Create a directory to save netCDF files
    save_dir = path_validate(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Parse file name and extract magnetic axis
    filename = path.stem
    magnetic_axis_r = float(filename.split("grid-")[1]) * 1.0e-2

    # Define progress bar
    progress = Progress()
    task_id = progress.add_task("Installing grid...", total=len(ZONES) + 1)

    # Open raw grid text data file
    with (
        progress,
        path.open(mode="r") as file,
    ):
        # parse grid coords for each zone
        for zone in ZONES:
            # parse grid resolution
            line = file.readline()
            L, M, N = [int(x) for x in line.split()]  # L: radial, M: poloidal, N: toroidal

            # number of table rows per r/z points
            num_rows = ceil(L * M / 6)

            # radial grid resolution is increased by 1 because of adding the magnetic axis point
            if zone in {"zone0", "zone11"}:
                L += 1

            # define grid array (4 dimensional array)
            grid = np.zeros((L, M, N, 2), dtype=np.float64)
            angles = np.zeros(N, dtype=np.float64)

            for n in range(N):
                # parse toroidal angle
                line = file.readline()
                toroidal_angle = float(line)

                # define (r, z) coords list at a polidal plane
                r_coords: list[float] = []
                z_coords: list[float] = []

                # parse r-z coords for each line
                for _ in range(num_rows):
                    line = file.readline()
                    r_coords += [float(x) * 1.0e-2 for x in line.split()]  # [cm] -> [m]

                for _ in range(num_rows):
                    line = file.readline()
                    z_coords += [float(x) * 1.0e-2 for x in line.split()]  # [cm] -> [m]
                line = file.readline()  # skip one line

                # add magnetic axis point coordinates
                if zone in {"zone0", "zone11"}:
                    index = 0
                    for _ in range(M):
                        r_coords.insert(index, magnetic_axis_r)
                        z_coords.insert(index, 0.0)
                        index += L

                # store coordinates into 4-D ndarray
                grid[:, :, n, 0] = np.reshape(r_coords, (L, M), order="F")
                grid[:, :, n, 1] = np.reshape(z_coords, (L, M), order="F")
                angles[n] = toroidal_angle

            # Define xarray dataset
            ds = xr.Dataset(
                data_vars=dict(
                    grid=(
                        ["ρ", "θ", "ζ", "RZ"],
                        grid,
                        dict(units="m", long_name="grid coordinates"),
                    ),
                ),
                coords=dict(
                    ρ=(
                        ["ρ"],
                        np.linspace(0, 1, L, endpoint=True),
                        dict(long_name="radial axis", description="normalized radial axis"),
                    ),
                    θ=(
                        ["θ"],
                        np.linspace(0, 1, M, endpoint=True),
                        dict(long_name="poloidal axis", description="normalized poloidal axis"),
                    ),
                    ζ=(
                        ["ζ"],
                        angles,
                        dict(units="deg", long_name="toroidal angle", description="toroidal angle"),
                    ),
                    RZ=(["RZ"], ["R", "Z"], dict(long_name="R-Z coordinates")),
                ),
                # Assign attributes
                attrs=dict(
                    num_cells=(L - 1) * (M - 1) * (N - 1),
                    description=f"grid data for {zone}",
                ),
            )

            # Save grid data
            ds.to_netcdf(save_dir / f"{filename}.nc", mode=mode, group=zone)

            # Advance progress
            progress.advance(task_id)

        # Save common variables for all zones
        ds = xr.Dataset(
            data_vars=dict(
                magnetic_axis=(
                    ["RZ"],
                    np.array([magnetic_axis_r, 0.0]),
                    dict(units="m", long_name="magnetic axis coordinates"),
                ),
            ),
            coords=dict(
                RZ=(["RZ"], ["R", "Z"], dict(long_name="R-Z coordinates")),
            ),
        )
        ds.to_netcdf(save_dir / f"{filename}.nc", mode=mode)

        # Advance progress
        progress.advance(task_id)


def install_indices(
    path: Path | str, grid_file: Path | str = PATH_TO_STORAGE / "emc3" / "grid-360.nc"
) -> None:
    """Install EMC3-EIRENE cell indices into netCDF file.

    There are many types of ways to apply an index into several cells of EMC3-EIRENE.
    We have three types of indices: physics, geometry, and coarse indices currently.

    The physics index is the index for use in the calculation of the physics quantity by EMC3-EIRENE.
    The geometry index is the index specifying each cell for several zones.
    The coarse index is the index for the coarse grid.

    Parameters
    ----------
    path : Path | str
        Path to the original text file written about physics cell indices (e.g. CELL_GEO).
    grid_file : Path | str, optional
        Path to the grid netCDF file, by default ``cherab/lhd/emc3/grid-360.nc`` under the user's
        cache directory.
    """
    path = exist_path_validate(path)

    # Define progress bar
    progress = Progress()
    task_id = progress.add_task("Installing indices...", total=len(ZONES) + 1)

    # Start Reading and Saving
    with progress:
        # Load cell index from text file starting from zero for c language index format
        indices_raw = np.loadtxt(path, dtype=np.uint32, skiprows=1) - 1

        # Define indicator to start reading indices
        idx_phys = 0
        idx_geo = 0
        idx_coarse = 0

        # Iterate over zones
        for zone in ZONES:
            # Open grid dataset
            ds_grid = xr.open_dataset(grid_file, group=zone)

            # Retrieve grid properties
            num_cells: int = ds_grid.attrs["num_cells"]
            L = ds_grid["ρ"].size
            M = ds_grid["θ"].size
            N = ds_grid["ζ"].size
            phis = ds_grid["ζ"].values

            ds_grid.close()

            data_arrays = {}

            # --------------------------------
            # === Get Physics cell indices ===
            # --------------------------------
            da, idx_phys = _get_physics_cell_indices(
                indices_raw, L, M, N, zone, num_cells, idx_phys
            )
            data_arrays["physics"] = da

            # -----------------------------------
            # === Get (Geometry) cell indices ===
            # -----------------------------------
            da, idx_geo = _get_geometry_cell_indices(L, M, N, zone, num_cells, idx_geo)
            if da is not None:
                data_arrays["cell"] = da

            # -------------------------------
            # === Get Coarse cell indices ===
            # -------------------------------
            da, idx_coarse = _get_coarse_indices(L, M, N, zone, idx_coarse)
            if da is not None:
                data_arrays["coarse"] = da

            # Create stored dataset
            ds = xr.Dataset(
                data_vars=data_arrays,
                coords=dict(
                    ρ=(
                        ["ρ"],
                        np.linspace(0, 1, L * 2 - 1, endpoint=True)[1::2],
                        dict(
                            long_name="radial axis",
                            description="normalized radial axis at the center of each cell",
                        ),
                    ),
                    θ=(
                        ["θ"],
                        np.linspace(0, 1, M * 2 - 1, endpoint=True)[1::2],
                        dict(
                            long_name="poloidal axis",
                            description="normalized poloidal axis at the center of each cell",
                        ),
                    ),
                    ζ=(
                        ["ζ"],
                        np.linspace(phis[0], phis[-1], phis.size * 2 - 1, endpoint=True)[1::2],
                        dict(
                            units="deg",
                            long_name="toroidal angle",
                            description="toroidal angle at the center of each cell",
                        ),
                    ),
                ),
                attrs=dict(
                    num_cells=num_cells,
                    description=f"Index for each cell in {zone}",
                ),
            )

            # Save
            ds.to_netcdf(grid_file, mode="a", group=f"{zone}/index")

            # Advance progress
            progress.advance(task_id)

        # Save number of cells data
        with path.open(mode="r") as file:
            num_total, num_plasma, num_vacuum = list(map(int, file.readline().split()))
        ds = xr.Dataset(
            attrs=dict(
                num_total=num_total,
                num_plasma=num_plasma,
                num_vacuum=num_vacuum,
            ),
        )
        ds.to_netcdf(grid_file, mode="a")
        progress.advance(task_id)


def _get_physics_cell_indices(
    indices_raw: np.ndarray,
    L: int,  # noqa: N803
    M: int,  # noqa: N803
    N: int,  # noqa: N803
    zone: str,
    num_cells: int,
    start: int,
) -> tuple[xr.DataArray, int]:
    """Return a Dataset containing the physical cell indices for the given zone.

    Parameters
    ----------
    indices_raw : np.ndarray
        Raw indices data.
    L : int
        Radial grid size.
    M : int
        Poloidal grid size.
    N : int
        Toroidal grid size.
    zone : str
        Zone name.
    num_cells : int
        Number of cells in the zone.
    start : int
        Start index of the indices data.

    Returns
    -------
    DataArray
        DataArray containing the physical cell indices.
    int
        Updated start index.
    """
    if zone in {"zone0", "zone11"}:
        L -= 1
        num_cells = (L - 1) * (M - 1) * (N - 1)
        # Extract indices for each zone and reshape it to 3-D array
        indices_temp = indices_raw[start : start + num_cells].reshape(
            (L - 1, M - 1, N - 1), order="F"
        )
        # Insert dummy indices for around magnetic axis region.
        # Inserted indices are duplicated from the first index of radial direction.
        indices = np.concatenate((indices_temp[0, ...][np.newaxis, :, :], indices_temp), axis=0)
        L += 1
        start += num_cells
    else:
        # Extract indices for each zone and reshape it to 3-D array
        indices = indices_raw[start : start + num_cells].reshape((L - 1, M - 1, N - 1), order="F")
        start += num_cells

    # Create stored dataset
    return xr.DataArray(
        data=indices,
        dims=["ρ", "θ", "ζ"],
        attrs=dict(
            long_name="physical cell index",
            description="Indices are defined for all zones.",
        ),
    ), start


def _get_geometry_cell_indices(
    L: int,  # noqa: N803
    M: int,  # noqa: N803
    N: int,  # noqa: N803
    zone: str,
    num_cells: int,
    start: int,
) -> tuple[xr.DataArray | None, int]:
    """Return a Dataset containing the geometry cell indices for the given zone.

    Parameters
    ----------
    L : int
        Radial grid size.
    M : int
        Poloidal grid size.
    N : int
        Toroidal grid size.
    zone : str
        Zone name.
    num_cells : int
        Number of cells in the zone.
    start : int
        Start index of the indices data.

    Returns
    -------
    DataArray | None
        DataArray containing the geometry cell indices.
        If the zone is not in the valid zone list, return None.
    int
        Updated start index.
    """
    if zone not in {
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
    }:
        return None, start

    indices = np.arange(start, start + num_cells, dtype=np.uint32).reshape(
        (L - 1, M - 1, N - 1), order="F"
    )
    start = 0  # Reset start index

    # Create stored dataset
    return xr.DataArray(
        data=indices,
        dims=["ρ", "θ", "ζ"],
        attrs=dict(
            long_name="geometry cell index",
            description="Indices are defined only for one zone. The index starts from 0.",
        ),
    ), start


def _get_coarse_indices(
    L: int,  # noqa: N803
    M: int,  # noqa: N803
    N: int,  # noqa: N803
    zone: str,
    start: int,
):
    if zone not in {
        "zone0",
        "zone11",
    }:
        return None, start

    boarders_radial = [17, 41, L]
    steps_radial = [1, 4, 4]
    step_poloidal = 6
    step_toroidal = 4

    indices_radial = (
        [i for i in range(0, boarders_radial[0], steps_radial[0])]
        + [i for i in range(boarders_radial[0], boarders_radial[1], steps_radial[1])]
        + [i for i in range(boarders_radial[1], boarders_radial[2], steps_radial[2])]
    )
    indices_poloidal = [i for i in range(0, M, step_poloidal)]
    indices_toroidal = [i for i in range(0, N, step_toroidal)]

    num_radial = len(indices_radial) - 1
    num_poloidal = len(indices_poloidal) - 1
    num_toroidal = len(indices_toroidal) - 1

    indices = np.zeros((L - 1, M - 1, N - 1), dtype=np.uint32)

    for i, j, k in np.ndindex(num_radial, num_poloidal, num_toroidal):
        indices[
            indices_radial[i] : indices_radial[i + 1],
            indices_poloidal[j] : indices_poloidal[j + 1],
            indices_toroidal[k] : indices_toroidal[k + 1],
        ] = i + j * num_radial + k * num_radial * num_poloidal + start

    # Create stored dataset
    return xr.DataArray(
        data=indices,
        dims=["ρ", "θ", "ζ"],
        attrs=dict(
            long_name="coarse cell index",
            description="Indices are defined only for one zone. The index starts from 0.",
            indices_radial=indices_radial,
            indices_poloidal=indices_poloidal,
            indices_toroidal=indices_toroidal,
        ),
    ), 0


def install_center_points(
    grid_file: Path | str = PATH_TO_STORAGE / "emc3" / "grid-360.nc",
) -> None:
    """Install EMC3-EIRENE center points into netCDF file.

    The center points are calculated from the grid data using the `.compute_centers` function.
    The center points are stored in the same netCDF file as a group named "centers".

    Parameters
    ----------
    grid_file : Path | str, optional
        Path to the grid netCDF file, by default ``cherab/lhd/emc3/grid-360.nc`` under the user's
        cache directory.
    """

    from ..grid import Grid

    # TODO: Implement the other zones
    zones = ["zone0", "zone11"]

    # TODO: Implement the other index types
    index_types = ["coarse", "cell"]

    # Define progress bar
    progress = Progress()
    task_id = progress.add_task("Installing center points...", total=len(zones) * len(index_types))

    with progress:
        for zone in zones:
            grid = Grid(zone, grid_file=grid_file)
            verts = grid.generate_vertices()
            cells = grid.generate_cell_indices()
            phis = grid.data_array["ζ"].values

            del grid

            for index_type in index_types:
                indices = xr.open_dataset(grid_file, group=f"{zone}/index")[index_type].data

                # Compute center points
                centers = compute_centers(verts, cells, indices)

                # Store center points
                ds = xr.Dataset(
                    data_vars=dict(
                        center_points=(
                            ["ρ", "θ", "ζ", "ΧΥΖ"],
                            centers,
                            dict(units="m", long_name="center points"),
                        )
                    ),
                    coords=dict(
                        ρ=(
                            ["ρ"],
                            np.linspace(0, 1, centers.shape[0] * 2 + 1, endpoint=True)[1::2],
                            dict(
                                long_name="radial axis",
                                description="normalized radial axis at each center point",
                            ),
                        ),
                        θ=(
                            ["θ"],
                            np.linspace(0, 1, centers.shape[1] * 2 + 1, endpoint=True)[1::2],
                            dict(
                                long_name="poloidal axis",
                                description="normalized poloidal axis at each center point",
                            ),
                        ),
                        ζ=(
                            ["ζ"],
                            np.linspace(phis[0], phis[-1], centers.shape[2] * 2 + 1, endpoint=True)[
                                1::2
                            ],
                            dict(
                                units="deg",
                                long_name="toroidal angle",
                                description="toroidal angle at each center point",
                            ),
                        ),
                        ΧΥΖ=(["ΧΥΖ"], ["X", "Y", "Z"], dict(long_name="X-Y-Z coordinates")),
                    ),
                    attrs=dict(
                        description=f"center points for {zone} with {index_type} index",
                    ),
                )
                ds.to_netcdf(grid_file, mode="a", group=f"{zone}/centers/{index_type}")

                # Advance progress
                progress.advance(task_id)


def install_data(
    directory_path: Path | str,
    grid_file: Path | str = PATH_TO_STORAGE / "emc3" / "grid-360.nc",
) -> None:
    """Install EMC3-EIRENE calculated data.

    `xarray.DataTree` is used to store the data into the same netCDF file as the grid data as a
    group named "data".

    So, the data tree structure looks like as follows:

    .. code-block:: none

        /
        ├── data
        :   ├── radiation
            │   ├── plasma
            │   ├── impurity
            │   └── total
            ├── density
            │   ├── electron
            │   ├── H+
            |   ├── C1+
            :   :
            |   └── Ne
            └── temperature
                ├── electron
                ├── ion
                ├── H
                └── H2

    For example, the radiation group contains one `xarray.Dataset` with three `xarray.DataArray`
    variables.
    Shared coordinates are stored in the data group.

    Parameters
    ----------
    directory_path : Path | str
        Path to the directory storing EMC3-calculated data.
    grid_file : Path | str, optional
        Path to the grid netCDF file, by default ``cherab/lhd/emc3/grid-360.nc`` under the user's
        cache directory.
    """
    # populate DataParser instance
    parser = DataParser(directory_path, grid_file)

    # Initialize data tree dictionary
    data_tree = {}

    # Define progress bar
    progress = Progress()
    task_id = progress.add_task("Installing physics data...", total=3)

    with progress:
        # ----------------------
        # === Load radiation ===
        # ----------------------
        radiations = {}
        for source in ["plasma", "impurity", "total"]:
            try:
                if source == "total":
                    radiations[source] = xr.DataArray(
                        parser.plasma_radiation(),
                        dims=["x"],
                        attrs=dict(units="W/m^3", long_name="total radiation"),
                    )
                else:
                    radiations[source] = xr.DataArray(
                        getattr(parser, f"{source}_radiation")(),
                        dims=["x"],
                        attrs=dict(units="W/m^3", long_name=f"{source} radiation"),
                    )

            except Exception as e:
                print(f"💥 Failed to install {source} raditaion: {e}")

        data_tree["data/radiation"] = xr.Dataset(
            data_vars=radiations,
            attrs=dict(description="radiation data"),
        )

        progress.advance(task_id)

        # --------------------
        # === Load density ===
        # --------------------
        densities = {}
        try:
            densities["electron"] = xr.DataArray(
                parser.density_electron(),
                dims=["x"],
                attrs=dict(units="1/m^3", long_name="electron density"),
            )
        except Exception as e:
            print(f"💥 Failed to install electron density: {e}")

        for source in ["ions", "neutrals"]:
            for atom, density in getattr(parser, f"density_{source}")().items():
                if len(density) == parser.num_plasma:
                    dim = ["x"]
                elif len(density) == parser.num_vacuum:
                    dim = ["v"]
                else:
                    raise ValueError("The length of the density data is not valid.")
                try:
                    densities[atom] = xr.DataArray(
                        density,
                        dims=dim,
                        attrs=dict(units="1/m^3", long_name=f"{atom} density"),
                    )
                except Exception as e:
                    print(f"💥 Failed to install {atom} density: {e}")

        data_tree["data/density"] = xr.Dataset(
            data_vars=densities,
            attrs=dict(description="density data"),
        )

        progress.advance(task_id)

        # ------------------------
        # === Load temperature ===
        # ------------------------
        temperatures = {}
        try:
            t_e, t_i = parser.temperature_electron_ion()
            temperatures["electron"] = xr.DataArray(
                t_e,
                dims=["x"],
                attrs=dict(units="eV", long_name="electron temperature"),
            )
            temperatures["ion"] = xr.DataArray(
                t_i,
                dims=["x"],
                attrs=dict(units="eV", long_name="ion temperature"),
            )
        except Exception as e:
            print(f"💥 Failed to install electron and ion temperature: {e}")

        for atom, temp in parser.temperature_neutrals().items():
            if len(temp) == parser.num_plasma:
                dim = ["x"]
            elif len(temp) == parser.num_vacuum:
                dim = ["v"]
            else:
                raise ValueError("The length of the temperature data is not valid.")
            try:
                temperatures[atom] = xr.DataArray(
                    temp,
                    dims=dim,
                    attrs=dict(units="eV", long_name=f"{atom} temperature"),
                )
            except Exception as e:
                print(f"💥 Failed to install {atom} temperature: {e}")

        data_tree["data/temperature"] = xr.Dataset(
            data_vars=temperatures,
            attrs=dict(description="temperature data"),
        )

        progress.advance(task_id)

        # Save data tree
        coords = {
            "data": xr.Dataset(
                coords=dict(
                    x=(
                        ["x"],
                        np.arange(parser.num_plasma),
                        dict(
                            long_name="Plasma cell index",
                            description="An number of this coords corresponds to the physics cell index",
                        ),
                    ),
                    v=(
                        ["v"],
                        np.arange(parser.num_vacuum),
                        dict(
                            long_name="Vacuum cell index",
                            description="An number of this coords corresponds to the physics cell index",
                        ),
                    ),
                )
            )
        }
        xr.DataTree.from_dict(coords | data_tree).to_netcdf(grid_file, mode="a")
