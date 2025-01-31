"""Module to fetch files from the remote server using the SFTP downloader."""

from __future__ import annotations

import os

import pooch
from pooch import SFTPDownloader
from rich.console import Console
from rich.table import Table

__all__ = ["fetch_file", "show_registries"]


HOSTNAME = os.environ.get("SSH_RAYTRACE_HOSTNAME", default="sftp://example.com/")
USERNAME = os.environ.get("SSH_RAYTRACE_USERNAME", default="username")
PASSWORD = os.environ.get("SSH_RAYTRACE_PASSWORD", default="password")

# Registry of the datasets
REGISTRIES = {
    # calcam calibration data
    "observer/IRVB.json": "a46010c1d67bbe71c0bf71c203447cb984c5622e63e10d20f5ee8b8d2cf93248",
    "observer/RB.json": "bffd237b98bc6a158fb4f4875d1d79d6571fe4770561f8e18fd1da72a63bcae2",
    # raysect machine mesh data
    "machine/divertor.rsm": "38335249c9bc38d0ddec8e40a575d10780202cee99e36476c8849450bb628ae0",
    "machine/plasma.rsm": "3198a722c28906bb5d59e77d190a0b91498ca5725ddf3bf7184e4441faf062d5",
    "machine/plasma_cut.rsm": "a91c6099e0be68e49e17202508aff4e6bb947756a143799a711c5dc27794cf8f",
    "machine/plates.rsm": "139a01f7615144adeae469e39f8b4c6e8727363b42fc30739a60457f944fc23c",
    "machine/port_65l.rsm": "443e5755d2cc159d588a8ea29e69293132b31eb7539d103e1ebe908058f6ec14",
    "machine/port_65u.rsm": "fa67308aac39c2dd09dfd8a5d9fc666568ab9f9d07af8970d3158d35992a2230",
    "machine/vessel.rsm": "4080d48e2af1365c8b1b52009b550de242815131a335530cf6061d727f24cca4",
    # machine wall outline data
    "machine/wall_outline.hdf5": "f82cdc636d6874a2a292ddf1d4e3077690d1c06b53c4d9fb7a7c210e358bf790",
    # material data
    "materials/sus316L.json": "0f473be0d3f9efc88d11d3cbacbc5a8596ff2b39dcb2dbd561d0fa116de5f301",
    # EMC3-EIRENE-related data
    "emc3.hdf5": "99fb503798bb80f207b9de70e9fa6605584020a4d0c6dbc49ecbf38792a3e7f4",
    "tetra/zone0.rsm": "2859a0618badd7d764691750bbc617479a00ee4f0a31dcf9d616d81174f3402b",
    "tetra/zone1.rsm": "a097e6f5dcf0319e3109fc0d7c010442297e6078778cbc795bc37729ff6fc2be",
    "tetra/zone2.rsm": "33f22ac916633281e7c246f1092b24e41e2b9ca9d4497793c2c87a349bad526a",
    "tetra/zone3.rsm": "0e9178decb85664309a5de2b6d0d1ea3df0866ecface573407c276f05547bad3",
    "tetra/zone4.rsm": "42c5c1469d9fd4acb14e5dbb581dda570804ce84f232c70178feb42cab0681cc",
    "tetra/zone11.rsm": "dc45ebe22b858db13f1cd73e43422629dde6adc2f8ca2a4c14cb8e4a2b83dd14",
    "tetra/zone12.rsm": "a16ef6d09ad1f7551280d405d91aa63021b5eac6e0e397e563ee3130ed52c98e",
    "tetra/zone13.rsm": "8ded151d42952c3b83b9293d9c3e4be28ebe847608d7e8707d8daf518063949f",
    "tetra/zone14.rsm": "3ab9bfd2e8c2b6631c269a02a1597deebde1fce973ab61f3db9065054ced5538",
    "tetra/zone15.rsm": "25381a62eda6c217f280e0bf8d93fe077bb1c517a1e3dfa1fc60043db5be5fa2",
}


def show_registries() -> None:
    """Show the registries of the datasets."""
    table = Table(title="Registries", show_lines=True)
    table.add_column("File Name", style="cyan", justify="left")
    table.add_column("SHA256", style="dim", justify="left")

    for name, sha256 in REGISTRIES.items():
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
        path=pooch.os_cache("cherab/lhd"),
        base_url=host,
        registry=REGISTRIES,
    )

    downloader = SFTPDownloader(
        username=username,
        password=password,
        progressbar=True,
        timeout=5,
    )
    return pup.fetch(name, downloader=downloader)
