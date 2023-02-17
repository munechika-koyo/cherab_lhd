"""Module to parse and load raw data calculated by EMC3-EIRENE."""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from .utility import DEFAULT_HDF5_PATH, exist_path_validate, path_validate

__all__ = ["DataParser"]


class DataParser:
    """EMC3-EIRENE raw data parser.

    This class offers methods to read several raw data calculated by EMC3-EIRENE.

    Parameters
    ----------
    directory_path
        path to the directory storing EMC3-calculated data.
    hdf5_path, optional
        path to the stored HDF5 file, by default ``~/.cherab/lhd/emc3.hdf5``.
    grid_group_name, optional
        name of grid group in the HDF5 file, by default ``grid-360``.

    Example
    -------
    .. prompt:: python >>> auto

        >>> from cherab.lhd.emc3.repository.parse import DataParser
        >>> parser = DataParser("path/to/the/data/directory")
        >>> parser.plasma_radiation()
        array([0.000e+00, 0.000e+00, 0.000e+00, ..., 9.450e-33, 4.922e-33,
               2.659e-33])
        >>>
        >>> parser.density_ions()
        {'H+': array([3.7917e+19, 3.5602e+19, 3.7012e+19, ..., 4.3626e+16, 1.5922e+16,
                      1.1246e+16]),
         'C1+': array([0.0000e+00, 0.0000e+00, 0.0000e+00, ..., 6.6688e+16, 6.2780e+16,
                       5.0213e+16]), ...
        }
    """

    def __init__(
        self,
        directory_path: Path | str,
        hdf5_path: Path | str = DEFAULT_HDF5_PATH,
        grid_group_name: str = "grid-360",
    ) -> None:
        # parameter validation
        self.directory_path = path_validate(directory_path)
        self.hdf5_path = exist_path_validate(hdf5_path)

        # compiled regular expressions for pattern matching
        self.pattern_index = re.compile(r"\s+\d+\n")
        self.pattern_value = re.compile(r" -?\d\.\d{4}E[+-]\d{2}")  # e.g.) -1.2345E+12

        # load number of plasma cells
        with h5py.File(self.hdf5_path, mode="r") as h5file:
            grid_group = h5file.get(grid_group_name)
            self.num_total = grid_group.attrs["num_total"]
            self.num_plasma = grid_group.attrs["num_plasma"]
            self.num_plasma_vac = grid_group.attrs["num_plasma_vac"]

    def plasma_radiation(self, filename: str = "RADIATION_1") -> NDArray[np.float64]:
        """Load plasma radation data.

        Parameters
        ----------
        filename
            loading text file name, by default ``"RADIATION_1"``.

        Returns
        -------
        numpy.ndarray
            1D plasma radiation data [W/m^3]
        """
        with open(self.directory_path / filename, "r") as f:
            _ = f.readline()  # skip first row
            radiation = f.read()
        # extract values
        radiation = re.findall(self.pattern_value, radiation)

        radiation = np.asarray_chkfinite(radiation, dtype=np.float64) * 1.0e6  # [W/cm^3] -> [W/m^3]

        # validation
        if radiation.size != self.num_plasma:
            raise ValueError(
                f"The size of radation data ({radiation.size}) "
                f"must be same as the number of plasma cells ({self.num_plasma})"
            )

        # make negative velues positive
        radiation[radiation < 0] = 0.0

        return radiation

    def impurity_radiation(self, filename: str = "IMP_RADIATION") -> NDArray[np.float64]:
        """Load impurity radation data.

        Parameters
        ----------
        filename
            loading text file name, by default ``"IMP_RADIATION"``.

        Returns
        -------
        numpy.ndarray
            1D impurity radiation data [W/m^3].
            negative values are made positive.
        """
        with open(self.directory_path / filename, "r") as f:
            radiation = f.read()
        # extract values
        radiation = re.findall(self.pattern_value, radiation)
        radiation = np.asarray_chkfinite(radiation, dtype=np.float64) * 1.0e6  # [W/cm^3] -> [W/m^3]

        # validation
        if radiation.size != self.num_plasma:
            raise ValueError(
                f"The size of impurity radation data ({radiation.size}) "
                f"must be same as the number of plasma cells ({self.num_plasma})"
            )

        # make negative values positive
        radiation = np.negative(radiation)

        # make negative values zero
        radiation[radiation < 0] = 0.0

        return radiation

    def radiation(self) -> NDArray[np.float64]:
        """Load total radation data: plasma + impurity radiation.

        Returns
        -------
        numpy.ndarray
            1D radiation data [W/m^3].
            negative values are made positive.
        """
        return self.plasma_radiation() + self.impurity_radiation()

    def density_electron(self, filename: str = "DENSITY") -> NDArray[np.float64]:
        """Load electron density data which is same as H+ ones.

        Parameters
        ----------
        filename
            loading text file name, by default ``"DENSITY"``.

        Returns
        -------
        numpy.ndarray
            1D electron density data [1/m^3]
        """
        with open(self.directory_path / filename, "r") as f:
            density = f.read()

        # divide values by index
        density = re.split(self.pattern_index, density)

        density = (
            np.asarray_chkfinite(re.findall(self.pattern_value, density[1]), dtype=np.float64)
            * 1.0e6
        )  # [1/cc] -> [1/m^3]

        # validation
        if density.size != self.num_plasma:
            raise ValueError(
                f"The size of electron density data ({density.size}) "
                f"must be same as the number of plasma cells ({self.num_plasma})"
            )

        return density

    def density_ions(self, filename: str = "DENSITY") -> dict[str, NDArray[np.float64]]:
        """Load ions density data. the list of species state is: H+, C1+,
        C2+,..., C+6, Ne1+,..., Ne10+. The total number is 17.

        Parameters
        ----------
        filename
            loading text file name, by default ``"DENSITY"``.

        Returns
        -------
        dict[str, numpy.ndarray]
            key is label of ion and value is density data [1/m^3]
        """
        state_list = ["H+"] + [f"C{i}+" for i in range(1, 7)] + [f"Ne{i}+" for i in range(1, 11)]
        density_ions = {}
        with open(self.directory_path / filename, "r") as f:
            densities = f.read()

        # divide values by index
        densities = re.split(self.pattern_index, densities)

        # mapping density values into each state lavel
        for ion, density in zip(state_list, densities[1:]):
            density_ions[ion] = (
                np.asarray_chkfinite(re.findall(self.pattern_value, density), dtype=np.float64)
                * 1.0e6
            )  # [1/cc] -> [1/m^3]
            # validation
            if density_ions[ion].size != self.num_plasma:
                raise ValueError(
                    f"The size of {ion} density data ({density_ions[ion].size}) "
                    f"must be same as the number of plasma cells ({self.num_plasma})"
                )

        return density_ions

    def density_neutrals(self) -> dict[str, NDArray[np.float64]]:
        """Load neutral particles density data. the list of species state is:
        H, H2, C, Ne.

        Returns
        -------
        dict[str, numpy.ndarray]
            key is label of species and value is density data [1/m^3]
        """
        atoms = ["H", "H2", "C", "Ne"]
        filenames = ["DENSITY_A", "DENSITY_M", "IMPURITY_NEUTRAL_2", "IMPURITY_NEUTRAL_3"]
        density_neutral = {}

        for filename, atom in zip(filenames, atoms):

            with open(self.directory_path / filename, "r") as f:
                density = f.read()

            density_neutral[atom] = (
                np.asarray_chkfinite(re.findall(self.pattern_value, density), dtype=np.float64)
                * 1.0e6
            )  # [1/cc] -> [1/m^3]

            # validation
            if (
                density_neutral[atom].size != self.num_plasma_vac
                and density_neutral[atom].size != self.num_plasma
            ):
                raise ValueError(
                    f"The size of {atom} density data ({density_neutral[atom].size}) "
                    f"must be same as the number of plasma ({self.num_plasma_vac}) with vacuume cells"
                )

        return density_neutral

    def temperature_electron_ion(
        self, filename: str = "TE_TI"
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Load electron and ion (H+) temperature.

        Parameters
        ----------
        filename
            loading text file name, by default ``"DENSITY"``.

        Returns
        -------
        tuple[numpy.ndarray]
            electron and ion temperature data [eV]
        """
        with open(self.directory_path / filename, "r") as f:
            temp = f.read().split()

        temp_e = np.asarray_chkfinite(temp[: self.num_plasma], dtype=np.float64)
        temp_ion = np.asarray_chkfinite(temp[self.num_plasma :], dtype=np.float64)

        # validation
        if temp_ion.size != self.num_plasma:
            raise ValueError(
                f"The size of ion temperature data ({temp_ion.size}) "
                f"must be same as the number of plasma cells ({self.num_plasma})"
            )

        return (temp_e, temp_ion)

    def temperature_neutrals(self) -> dict[str, NDArray[np.float64]]:
        """Load neutral particles temperature data. the list of species state
        is: H, H2.

        Returns
        -------
        dict[str, numpy.ndarray]
            key is label of species and value is temperature data [eV]
        """
        atoms = ["H", "H2"]
        filenames = ["TEMPERATURE_A", "TEMPERATURE_M"]
        temperature_neutral = {}

        for filename, atom in zip(filenames, atoms):

            with open(self.directory_path / filename, "r") as f:
                temperature = f.read()

            temperature_neutral[atom] = np.asarray_chkfinite(
                re.findall(self.pattern_value, temperature), dtype=np.float64
            )

            # validation
            if temperature_neutral[atom].size != self.num_plasma_vac:
                raise ValueError(
                    f"The size of {atom} electron data ({temperature_neutral[atom].size}) "
                    f"must be same as the number of plasma with vacuume cells ({self.num_plasma_vac})"
                )

        return temperature_neutral
