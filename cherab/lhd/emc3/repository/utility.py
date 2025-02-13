"""Utilities for managing the local repository."""

from __future__ import annotations

from pathlib import Path

__all__ = ["exist_path_validate", "path_validate"]


def exist_path_validate(path: Path | str) -> Path:
    """Validate exist path and return :obj:`~pathlib.Path` instance.

    This checks the path with or w/o ``.txt`` suffix and return the
    existing one.

    Parameters
    ----------
    path : Path | str
        Path to a text file.

    Returns
    -------
    Path
        Path to the existing text file.
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
    path : Path | str
        Arbitrary path.

    Returns
    -------
    Path
        Path to the file or directory.
    """
    if isinstance(path, (Path, str)):
        path = Path(path)
    else:
        raise TypeError(f"{path} must be string or pathlib.Path instance.")
    return path
