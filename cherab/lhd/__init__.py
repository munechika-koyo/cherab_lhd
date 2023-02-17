from pathlib import Path

# parse the package version number
with open(Path(__file__).parent.absolute() / "VERSION") as _f:
    __version__ = _f.read().strip()
