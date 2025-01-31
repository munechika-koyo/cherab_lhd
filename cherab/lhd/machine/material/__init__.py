"""This subpackage provides material definitions for the LHD machine."""

from .metal import SUS316L
from .roughmetal import RoughSUS316L

__all__ = ["RoughSUS316L", "SUS316L"]
