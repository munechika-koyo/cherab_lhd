"""Module to provide useful functions around Installing EMC3-related data."""

from __future__ import annotations

import re
from math import ceil
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from ...tools.fetch import PATH_TO_STORAGE, fetch_file
from ...tools.spinner import Spinner
from .parse import DataParser
from .utility import exist_path_validate, path_validate

__all__ = ["install_grids", "install_cell_indices", "install_physical_cell_indices", "install_data"]


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

    # Open raw grid text data file
    with (
        Spinner(text=f"install grid data from {path.name}...") as sp,
        path.open(mode="r") as file,
    ):
        # Load each zone area
        zones = [f"zone{i}" for i in range(0, 21 + 1)]

        # parse grid coords for each zone
        for zone in zones:
            # Define stored dataset
            ds = xr.Dataset()

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

                # store coordinates into 4-D ndarray)
                grid[:, :, n, 0] = np.reshape(r_coords, (L, M), order="F")
                grid[:, :, n, 1] = np.reshape(z_coords, (L, M), order="F")
                angles[n] = toroidal_angle

            # Assign values and coords
            ds = ds.assign(
                dict(
                    grid=(
                        ["rho", "theta", "zeta", "RZ"],
                        grid,
                        dict(units="m", long_name="grid coordinates"),
                    ),
                    angles=(
                        ["zeta"],
                        angles,
                        dict(units="deg", long_name="toroidal angle"),
                    ),
                ),
            )
            ds = ds.assign_coords(
                rho=(["rho"], np.arange(L), dict(long_name="radial index")),
                theta=(["theta"], np.arange(M), dict(long_name="poloidal index")),
                zeta=(["zeta"], np.arange(N), dict(long_name="toroidal index")),
                RZ=(["RZ"], ["R", "Z"], dict(long_name="R-Z coordinates")),
            )

            # Assign attributes
            ds = ds.assign_attrs(
                dict(num_cells=(L - 1) * (M - 1) * (N - 1), description=f"grid data for {zone}"),
            )

            # save grid data
            ds.to_netcdf(save_dir / f"{filename}.nc", mode=mode, group=zone)

        # save common variables for all zones
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

        sp.ok()


def install_physical_cell_indices(
    path: Path | str,
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    grid_group_name: str = "grid-360",
    update: bool = False,
) -> None:
    """Reconstruct physical cell indices and install it to a HDF5 file.

    EMC3-EIRENE has numerous geometric cells, each of which forms a cubic-like shape with 8 vetices
    in each zones. Several integrated cells, what is called ``physical cell``, are defined where the
    same physical quantities are calculated.
    Here such indices are parsed from text file and save them into an HDF5 file in each zone group.

    Parameters
    ----------
    path : Path | str
        Path to the raw text file: e.g. ``CELL_GEO``.
    hdf5_path : Path | str, optional
        Path to the stored HDF5 file, by default `DEFAULT_HDF5_PATH`.
    grid_group_name : str, optional
        Name of grid group in the HDF5 file, by default ``grid-360``.
    update : bool, optional
        Whether or not to update/override dataset, by default False.
    """
    # validate parameters
    path = exist_path_validate(path)
    hdf5_path = path_validate(hdf5_path)

    # start spinner and open HDF5 file
    with (
        Spinner(text=f"install physical cell indices data from {path.stem}...") as sp,
        h5py.File(hdf5_path, mode="r+") as h5file,
    ):
        # obtain grid group
        grid_group = h5file.get(grid_group_name)
        if grid_group is None:
            raise ValueError(f"{grid_group_name} does not exist in {hdf5_path}.")

        # Load cell index from text file starting from zero for c language index format
        indices_raw = np.loadtxt(path, dtype=np.uint32, skiprows=1) - 1

        # extract and sort zone keys
        zones = [key for key in grid_group.keys() if "zone" in key]
        zones = sorted(zones, key=lambda s: int(re.search(r"\d+", s).group()))

        start = 0
        for zone in zones:
            zone_group = grid_group.get(zone)
            num_cells: int = zone_group["grids"].attrs["num_cells"]
            L: int = zone_group["grids"].attrs["L"]
            M: int = zone_group["grids"].attrs["M"]
            N: int = zone_group["grids"].attrs["N"]

            # create index group
            if "index" not in zone_group:
                index_group = zone_group.create_group("index")
            else:
                index_group = zone_group.get("index")

            if zone in {"zone0", "zone11"}:
                L -= 1
                num_cells = (L - 1) * (M - 1) * (N - 1)

                # extract indices for each zone and reshape it to 3-D array
                indices_temp = indices_raw[start : start + num_cells].reshape(
                    (L - 1, M - 1, N - 1), order="F"
                )

                # insert dummy indices for around magnetic axis region.
                # inserted indices are duplicated from the first index of radial direction.
                indices = np.concatenate(
                    (indices_temp[0, ...][np.newaxis, :, :], indices_temp), axis=0
                )
                L += 1
                start += num_cells
            else:
                # extract indices for each zone and reshape it to 3-D array
                indices = indices_raw[start : start + num_cells].reshape(
                    (L - 1, M - 1, N - 1), order="F"
                )
                start += num_cells

            if update is True and "physics" in index_group:
                sp.write(f"update {index_group.name}/physics")
                del index_group["physics"]
            ds = index_group.create_dataset(name="physics", data=indices)

            # save attribution information
            ds.attrs["description"] = "for EMC3-calculated data"
            ds.attrs["shape description"] = "radial index, poloidal index, toroidal index"
            ds.attrs["L"] = L - 1
            ds.attrs["M"] = M - 1
            ds.attrs["N"] = N - 1

        # save number of cells data
        with path.open(mode="r") as file:
            num_total, num_plasma, num_plasma_vac = list(map(int, file.readline().split()))
        grid_group.attrs["num_total"] = num_total
        grid_group.attrs["num_plasma"] = num_plasma
        grid_group.attrs["num_plasma_vac"] = num_plasma_vac

        sp.ok()


def install_cell_indices(
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    grid_group_name: str = "grid-360",
    update: bool = False,
) -> None:
    """Create EMC3-EIRENE geometry cell indices and install it to a HDF5 file.

    EMC3-EIRENE has numerous geometric cells, each of which forms a cubic-like shape with 8 vetices
    in each zones.
    To identify each cell, an index number is allocated to each cell.
    Each number of an index is uneaque in all zones where plasma exists, so targeted zone labels are
    ``zone0`` - ``zone4`` and ``zone11`` - ``zone15``.
    The index data is saved into an HDF5 file in each zone group.

    Parameters
    ----------
    hdf5_path : Path | str, optional
        Path to the stored HDF5 file, by default `DEFAULT_HDF5_PATH`.
    grid_group_name : str, optional
        Name of grid group in the HDF5 file, by default ``grid-360``.
    update : bool, optional
        Whether or not to update/override dataset, by default False.
    """
    # validate parameters
    hdf5_path = path_validate(hdf5_path)

    # start spinner and open HDF5 file
    with (
        Spinner(text="create and install cell indices...") as sp,
        h5py.File(hdf5_path, mode="r+") as h5file,
    ):
        # obtain grid group
        grid_group = h5file.get(grid_group_name)
        if grid_group is None:
            raise ValueError(f"{grid_group_name} does not exist in {hdf5_path}.")

        # define zones
        zones = [
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

        start = 0
        for zone in zones:
            zone_group = grid_group.get(zone)
            num_cells: int = zone_group["grids"].attrs["num_cells"]
            L: int = zone_group["grids"].attrs["L"]
            M: int = zone_group["grids"].attrs["M"]
            N: int = zone_group["grids"].attrs["N"]

            # create index group
            if "index" not in zone_group:
                index_group = zone_group.create_group("index")
            else:
                index_group = zone_group.get("index")

            # create cell index array
            indices = np.arange(start, start + num_cells, dtype=np.uint32).reshape(
                (L - 1, M - 1, N - 1), order="F"
            )

            if update is True and "cell" in index_group:
                sp.write(f"update {index_group.name}/cell")
                del index_group["cell"]
            ds = index_group.create_dataset(name="cell", data=indices)

            # save attribution information
            ds.attrs["description"] = "fine cell index"
            ds.attrs["shape description"] = "radial index, poloidal index, toroidal index"
            ds.attrs["L"] = L - 1
            ds.attrs["M"] = M - 1
            ds.attrs["N"] = N - 1

        sp.ok()


def install_data(
    directory_path: Path | str,
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    grid_group_name: str = "grid-360",
) -> None:
    """Install EMC3-EIRENE calculated data into a HDF5 file.

    Parameters
    ----------
    directory_path : Path | str
        Path to the directory storing EMC3-calculated data.
    hdf5_path : Path | str, optional
        Path to the stored HDF5 file, by default `DEFAULT_HDF5_PATH`.
    grid_group_name : str, optional
        Name of grid group in the HDF5 file, by default ``grid-360``.
    """
    # populate DataParser instance
    parser = DataParser(
        directory_path=directory_path, hdf5_path=hdf5_path, grid_group_name=grid_group_name
    )
    # start spinner
    with (
        Spinner(text=f"install calculated data from {directory_path}...") as sp,
        h5py.File(hdf5_path, mode="r+") as h5file,
    ):
        data_group = h5file[grid_group_name].create_group("data")

        # radiation
        for source in ["plasma", "impurity", "total"]:
            try:
                if source == "total":
                    data_group.create_dataset(name=f"radiation/{source}", data=parser.radiation())
                else:
                    data_group.create_dataset(
                        name=f"radiation/{source}", data=getattr(parser, f"{source}_radiation")()
                    )
                sp.write(f"✅ {source} radiation was installed")
            except Exception:
                sp.write(f"💥 Failed to install {source} raditaion")
        data_group["radiation"].attrs["unit"] = "W/m^3"

        # density
        try:
            data_group.create_dataset(name="density/electron", data=parser.density_electron())
            sp.write("✅ electron density was installed")
        except Exception:
            sp.write("💥 Failed to install electron density")

        for source in ["ions", "neutrals"]:
            for atom, density in getattr(parser, f"density_{source}")().items():
                try:
                    data_group.create_dataset(name=f"density/{atom}", data=density)
                    sp.write(f"✅ {atom} density was installed.")
                except Exception:
                    sp.write(f"💥 Failed to install {atom} density")
        data_group["density"].attrs["unit"] = "1/m^3"

        # temperature
        try:
            t_e, t_i = parser.temperature_electron_ion()
            data_group.create_dataset(name="temperature/electron", data=t_e)
            data_group.create_dataset(name="temperature/ion", data=t_i)
            sp.write("✅ electron and ion temperature was installed.")
        except Exception:
            sp.write("💥 Failed to install electron and ion temperature")

        for atom, temp in parser.temperature_neutrals().items():
            try:
                data_group.create_dataset(name=f"temperature/{atom}", data=temp)
                sp.write(f"✅ {atom} temperature was installed.")
            except Exception:
                sp.write(f"💥 Failed to install {atom} temperature")
        data_group["temperature"].attrs["unit"] = "eV"
