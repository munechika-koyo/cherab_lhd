"""Module to fetch files from the remote server using the SFTP downloader."""

from __future__ import annotations

import json
import os
from importlib.resources import files

import pooch
from pooch import SFTPDownloader
from rich.console import Console
from rich.table import Table

__all__ = ["fetch_file", "show_registries", "PATH_TO_STORAGE"]


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
        Registries of the datasets, where key is the file name and value is the SHA256 hash.
    """
    with files("cherab.lhd.tools").joinpath("registries.json").open("r") as file:
        return json.load(file)


def show_registries() -> None:
    """Show the registries of the datasets."""
    table = Table(title="Registries", show_lines=True)
    table.add_column("File Name", style="cyan", justify="left")
    table.add_column("SHA256", style="dim", justify="left")

    for name, sha256 in get_registries().items():
        table.add_row(name, sha256)
    console = Console()
    console.print(table)


def fetch_file(
    name: str,
    host: str = HOSTNAME,
    username: str = USERNAME,
    password: str = PASSWORD,
) -> str:
    """Fetch the file from the remote server using the configured SFTP downloader.

    Fetched data will be stored in the cache directory like `~/.cache/cherab/lhd`.

    Parameters
    ----------
    name : str
        Name of the file to fetch.
    host : str, optional
        Host name of the server, by default ``sftp://example.com/``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_HOSTNAME`.
        Host name should be in the format ``sftp://{host's name or ip}/{directories}``.
    username : str, optional
        Username to authenticate with the server, by default ``username``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_USERNAME`.
    password : str, optional
        Password to authenticate with the server, by default ``password``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_PASSWORD`.

    Returns
    -------
    str
        Path to the fetched file.
    """
    pup = pooch.create(
        path=PATH_TO_STORAGE,
        base_url=host,
        registry=get_registries(),
    )

    downloader = SFTPDownloader(
        username=username,
        password=password,
        progressbar=True,
        timeout=5,
    )
    return pup.fetch(name, downloader=downloader)
