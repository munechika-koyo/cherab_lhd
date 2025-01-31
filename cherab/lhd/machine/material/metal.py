"""Module defining metal material classes."""

import json

from numpy import array
from raysect.optical import InterpolatedSF
from raysect.optical.material import Conductor

from ...tools.fetch import fetch_file

__all__ = ["SUS316L"]


class _DataLoader(Conductor):
    def __init__(self, filename):
        path = fetch_file(f"materials/{filename}.json")
        with open(path, "r") as f:
            data = json.load(f)

        wavelength = array(data["wavelength"])
        index = InterpolatedSF(wavelength, array(data["index"]))
        extinction = InterpolatedSF(wavelength, array(data["extinction"]))

        super().__init__(index, extinction)


class SUS316L(_DataLoader):
    """Stainless Used Steel 316L metal material."""

    def __init__(self):
        super().__init__("sus316L")
