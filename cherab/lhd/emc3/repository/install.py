"""Module to provide useful functions around Install EMC3-related data."""

from __future__ import annotations

import re
from math import ceil
from pathlib import Path

import h5py
import numpy as np

from ...tools.spinner import Spinner
from .parse import DataParser
from .utility import DEFAULT_HDF5_PATH, exist_path_validate, path_validate

__all__ = ["install_grids", "install_cell_indices", "install_physical_cell_indices", "install_data"]


def install_grids(
    path: Path | str,
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    update: bool = False,
) -> None:
    """Install EMC3-EIRENE grid data into a HDF5 file.

    Default EMC3 grid data is written in a text file, so they should be relocated into
    the HDF5 file.
    The value of :math:`R` coordinates of the magnetic axis is parsed from the filename of
    the text fil. e.g. ``grid-360.text`` means $R = 3.6$ m. :math:`Z` is always regarded as 0.

    Parameters
    ----------
    path : Path | str
        Path to the original text file written about grid coordinates at each zone.
    hdf5_path : Path | str, optional
        Path to the stored HDF5 file, by default `DEFAULT_HDF5_PATH`.
    update : bool, optional
        Whether or not to update/override dataset, by default False.
    """
    # validate paths
    path = exist_path_validate(path)
    hdf5_path = path_validate(hdf5_path)

    # parse file name and extract magnetic axis
    filename = path.stem
    magnetic_axis_r = float(filename.split("grid-")[1]) * 1.0e-2

    # start spinner and open HDF5 and raw grid data file
    with (
        Spinner(text=f"install grid data from {path.name}...") as sp,
        h5py.File(hdf5_path, mode="w") as h5file,
        path.open(mode="r") as file,
    ):
        # create grid group
        grid_group = h5file.create_group(f"{path.stem}")

        # Load each zone area
        zones = [f"zone{i}" for i in range(0, 21 + 1)]

        # parse grid coords for each zone
        for zone in zones:
            # create zone group
            zone_group = grid_group.create_group(zone)

            # parse grid resolution
            line = file.readline()
            L, M, N = [int(x) for x in line.split()]  # L: radial, M: poloidal, N: toroidal

            # number of table rows per r/z points
            num_rows = ceil(L * M / 6)

            # radial grid resolution is increased by 1 because of adding the magnetic axis point
            if zone in {"zone0", "zone11"}:
                L += 1

            # define grid array (4 dimension array)
            grid = np.zeros((L, M, N, 3), dtype=np.float64)

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
                grid[:, :, n, 2] = np.full((L, M), toroidal_angle)

            # save grid data as dataset
            if update is True and "grids" in zone_group:
                sp.write(f"update {zone_group.name}/grids")
                del zone_group["grids"]
            dset = zone_group.create_dataset(name="grids", data=grid)

            # store grid config
            dset.attrs["L"] = L
            dset.attrs["M"] = M
            dset.attrs["N"] = N
            dset.attrs["num_cells"] = (L - 1) * (M - 1) * (N - 1)
            dset.attrs["shape description"] = (
                "radial index, poloidal index, toroidal index, (r, z, phi) coordinates"
            )

        # add attribution
        grid_group.attrs["magnetic axis (R, Z) [m]"] = (magnetic_axis_r, 0)

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
