"""Utilities for managing the local repository."""

from __future__ import annotations

from pathlib import Path

from ...tools.fetch import PATH_TO_STORAGE

__all__ = ["DEFAULT_HDF5_PATH", "DEFAULT_TETRA_MESH_PATH", "exist_path_validate", "path_validate"]

DEFAULT_HDF5_PATH = PATH_TO_STORAGE / "emc3.hdf5"
DEFAULT_TETRA_MESH_PATH = PATH_TO_STORAGE / "tetra/"


def exist_path_validate(path: Path | str) -> Path:
    """Validate exist path and return :obj:`~pathlib.Path` instance.

    This checks the path with or w/o ``.txt`` suffix and return the
    existing one.

    Parameters
    ----------
    path
        path to a text file
    """
    path = path_validate(path)
    if not path.exists():
        path = path.with_suffix(".txt")
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

    return path


def path_validate(path: Path | str) -> Path:
    """Validate path and return :obj:`~pathlib.Path` instance.

    Parameters
    ----------
    path
        arbitarary path
    """
    if isinstance(path, (Path, str)):
        path = Path(path)
    else:
        raise TypeError(f"{path} must be string or pathlib.Path instance.")
    return path
