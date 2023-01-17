"""Utilities for managing the local repository."""
from pathlib import Path

__all__ = ["DEFAULT_HDF5_PATH", "exist_path_validate", "path_validate"]

DEFAULT_HDF5_PATH = Path("~/.cherab/lhd/emc3.hdf5").expanduser()


def exist_path_validate(path: Path | str) -> Path:
    """Validate exist path and return :obj:`~pathlib.Path` instance.

    This checks the path with or w/o ``.txt`` suffix and return the
    existing one.
    """
    path = path_validate(path)
    if not path.exists():
        path = path.with_suffix(".txt")
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

    return path


def path_validate(path: Path | str) -> Path:
    """Validate path and return :obj:`~pathlib.Path` instance."""
    if isinstance(path, (Path, str)):
        path = Path(path)
    else:
        raise TypeError(f"{path} must be string or pathlib.Path instance.")
    return path
