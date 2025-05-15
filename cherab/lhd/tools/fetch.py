"""Module to fetch files from the remote server using the SFTP downloader."""

from __future__ import annotations

import json
import os
from importlib.resources import files

import pooch
from pooch import SFTPDownloader
from rich.console import Console
from rich.style import Style
from rich.table import Table

__all__ = ["fetch_file", "get_registries", "get_urls", "show_registries", "PATH_TO_STORAGE"]


HOSTNAME = os.environ.get("SSH_RAYTRACE_HOSTNAME", default="sftp://example.com/")
USERNAME = os.environ.get("SSH_RAYTRACE_USERNAME", default="username")
PASSWORD = os.environ.get("SSH_RAYTRACE_PASSWORD", default="password")

# Path to the storage directory
PATH_TO_STORAGE = pooch.os_cache("cherab/lhd")


def get_registries() -> dict[str, str]:
    """Get the registries of the datasets.

    Returns
    -------
    dict[str, str]
        Registries of the datasets, where key is the file name and value is the hash.

    Examples
    --------
    >>> get_registries()
    {
        "emc3/grid-360.nc": "...",
        "machine/divertor.rsm": "...",
        ...
    }
    """
    with files("cherab.lhd.tools").joinpath("registries.json").open("r") as file:
        return json.load(file)


def get_urls() -> dict[str, str]:
    """Get the URLs of the datasets.

    Some datasets are available through the specified URLs, which are stored in a JSON file.
    Datasets with no URL specified are downloaded from the address specified by `.fetch_file`'s
    `host` parameter.

    Returns
    -------
    dict[str, str]
        URLs of the datasets, where key is the file name and value is the URL.

    Examples
    --------
    >>> get_urls()
    {
        "grid-demo.nc": "doi:10.5281/zenodo.1234567/grid-demo.nc",
        ...
    }
    """
    with files("cherab.lhd.tools").joinpath("urls.json").open("r") as file:
        return json.load(file)


def show_registries() -> None:
    """Show the registries of the datasets."""
    table = Table(title="Registries", show_lines=True)
    table.add_column("File Name", style="cyan", justify="left")
    table.add_column("Hash Value", style="dim", justify="left")

    for name, sha256 in get_registries().items():
        table.add_row(name, sha256)
    console = Console(style=Style(bgcolor="black"))
    console.print(table)


def fetch_file(
    name: str,
    host: str = HOSTNAME,
    username: str = USERNAME,
    password: str = PASSWORD,
) -> str:
    """Fetch the file remotely.

    Fetched data will be stored in the cache directory like `~/.cache/cherab/lhd`.
    The download destination (URL) is specified by `host` parameter, or obtained by `.get_urls`
    function for each file.

    Parameters
    ----------
    name : str
        Name of the file to fetch.
    host : str, optional
        Host name of the server, by default ``sftp://example.com/``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_HOSTNAME`.
        Host name should be in the format
        ``{protocol (e.g. sftp)}://{host's name or ip}/{directories}``.
    username : str, optional
        Username to authenticate with the sftp server, by default ``username``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_USERNAME`.
    password : str, optional
        Password to authenticate with the sftp server, by default ``password``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_PASSWORD`.

    Returns
    -------
    str
        Path to the fetched file.
    """
    # Check if the name is in the registries
    registries = get_registries()
    if name not in registries:
        raise ValueError(f"File {name} is not in the registries.")

    pup = pooch.create(
        path=PATH_TO_STORAGE,
        base_url=host,
        registry=registries,
        urls=get_urls(),
    )

    if pup.get_url(name).startswith("sftp://"):
        downloader = SFTPDownloader(
            username=username,
            password=password,
            progressbar=True,
            timeout=5,
        )
    else:
        # If the URL does not start with sftp://, use the default downloader
        downloader = None

    return pup.fetch(name, downloader=downloader)
