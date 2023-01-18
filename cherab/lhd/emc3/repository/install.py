"""Module to provide useful functions around Install EMC3-related data."""
from __future__ import annotations

from math import ceil
from pathlib import Path

import h5py
import numpy as np

from ...tools.spinner import Spinner
from .parse import DataParser
from .utility import DEFAULT_HDF5_PATH, exist_path_validate, path_validate

__all__ = ["install_grids", "install_cell_index"]


def install_grids(
    path: Path | str,
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
) -> None:
    """Install EMC3-EIRENE grid data into a HDF5 file.

    Default EMC3 grid data is written in a text file, so they should be relocated into
    the HDF5 file.
    The value of :math:`R` coordinates of the magnetic axis is parsed from the filename of
    the text fil. e.g. ``grid-360.text`` means $R = 3.6$ m. :math:`Z` is always regarded as 0.

    Parameters
    ----------
    path
        path to the original text file written about grid coordinates at each zone.
    hdf5_path, optional
        path to the stored HDF5 file, by default ``~/.cherab/lhd/emc3.hdf5``.
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

            # parse number of grid resolution
            line = file.readline()
            L, M, N = [int(x) for x in line.split()]

            # number of table rows per r/z points
            num_rows = ceil(L * M / 6)

            # radial grid resolution is increased by 1 because of adding magnetic axis point
            if zone in {"zone0", "zone11"}:
                L += 1

            # define grid array
            grid = np.zeros((L * M, 3, N), dtype=np.float64)

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

                # store coordinates into 3-D ndarray (r, z, phi)
                grid[:, 0, n] = r_coords
                grid[:, 1, n] = z_coords
                grid[:, 2, n] = np.repeat(toroidal_angle, L * M)

            # save grid data as dataset
            dset = zone_group.create_dataset(name="grids", data=grid)

            # store grid config
            dset.attrs["L"] = L
            dset.attrs["M"] = M
            dset.attrs["N"] = N
            dset.attrs["num_cells"] = (L - 1) * (M - 1) * (N - 1)

        # add attribution
        grid_group.attrs["magnetic axis (R, Z) [m]"] = (magnetic_axis_r, 0)

        sp.ok()


def install_cell_index(
    path: Path | str,
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    grid_group_name: str = "grid-360",
) -> None:
    """Reconstruct physical cell indices and install it to a HDF5 file.

    EMC3-EIRENE has numerous geometric cells, each of which forms a cubic-like shape with 8 vetices
    in each zones.
    To identify each cell, an index number called 'physical index' is allocated to each cell.
    Here such indices are parsed from text file and save them into an HDF5 file in each zone group.

    Parameters
    ----------
    path
        path to the raw text file: e.g. ``CELL_GEO``.
    hdf5_path, optional
        path to the stored HDF5 file, by default ``~/.cherab/lhd/emc3.hdf5``.
    grid_group_name, optional
        name of grid group in the HDF5 file, by default ``grid-360``.
    """
    # validate parameters
    path = exist_path_validate(path)
    hdf5_path = path_validate(hdf5_path)

    # start spinner and open HDF5 file
    with (
        Spinner(text=f"install cell index data from {path.stem}...") as sp,
        h5py.File(hdf5_path, mode="r+") as h5file,
    ):
        # obtain grid group
        grid_group = h5file.get(grid_group_name)
        if grid_group is None:
            raise ValueError(f"{grid_group_name} does not exist in {hdf5_path}.")

        # Load cell index from text file starting from zero for c language index format
        indices = np.loadtxt(path, dtype=np.uint32, skiprows=1) - 1

        start = 0
        for zone in grid_group.keys():
            zone_group = grid_group[zone]
            num_cells: int = zone_group["grids"].attrs["num_cells"]
            L: int = zone_group["grids"].attrs["L"]
            M: int = zone_group["grids"].attrs["M"]
            N: int = zone_group["grids"].attrs["N"]
            if zone in {"zone0", "zone11"}:
                L -= 1
                num_cells = (L - 1) * (M - 1) * (N - 1)
                index_array = indices[start : start + num_cells]
                values = np.zeros((N - 1) * (M - 1))
                insert_indices = np.zeros((N - 1) * (M - 1), dtype=int)
                j = 0
                for n in range(N - 1):
                    for m in range(M - 1):
                        i = n * (M - 1) * (L - 1) + m * (L - 1)
                        insert_indices[j] = i
                        values[j] = index_array[i]
                        j += 1

                index_array = np.insert(index_array, insert_indices, values)
                start += num_cells
            else:
                index_array = indices[start : start + num_cells]
                start += num_cells

            dset = zone_group.create_dataset(name="index", data=index_array)

        # save number of cells data
        with path.open(mode="r") as file:
            num_total, num_plasma, num_plasma_vac = list(map(int, file.readline().split()))
        grid_group.attrs["num_total"] = num_total
        grid_group.attrs["num_plasma"] = num_plasma
        grid_group.attrs["num_plasma_vac"] = num_plasma_vac

        sp.ok()


def install_data(
    directory_path: Path | str,
    hdf5_path: Path | str = DEFAULT_HDF5_PATH,
    grid_group_name: str = "grid-360",
) -> None:
    """Install EMC3-EIRENE calculated data into a HDF5 file.

    Parameters
    ----------
    directory_path
        path to the directory storing EMC3-calculated data.
    hdf5_path, optional
        path to the stored HDF5 file, by default ``~/.cherab/lhd/emc3.hdf5``.
    grid_group_name, optional
        name of grid group in the HDF5 file, by default ``grid-360``.
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
                    data_group.create_dataset(
                        name=f"radiation/{source}", data=getattr(parser, f"radiation")()
                    )
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

        # temerature
        try:
            te, ti = parser.temperature_electron_ion()
            data_group.create_dataset(name=f"temperature/electron", data=te)
            data_group.create_dataset(name=f"temperature/ion", data=ti)
            sp.write(f"✅ electron and ion temperature was installed.")
        except Exception:
            sp.write("💥 Failed to install electron and ion temperature")

        for atom, temp in parser.temperature_neutrals().items():
            try:
                data_group.create_dataset(name=f"temperature/{atom}", data=temp)
                sp.write(f"✅ {atom} temperature was installed.")
            except Exception:
                sp.write(f"💥 Failed to install {atom} temperature")
        data_group["temperature"].attrs["unit"] = "eV"
