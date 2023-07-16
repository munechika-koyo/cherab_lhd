"""Developper CLI: building (meson) sources in place, document, etc."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import rich_click as click

try:
    import tomllib
except ImportError:
    import tomli as tomllib


BASE_DIR = Path(__file__).parent.absolute()
BUILD_DIR = BASE_DIR / "build"
SRC_PATH = BASE_DIR / "cherab"
DOC_ROOT = BASE_DIR / "docs"
ENVS = dict(os.environ)
N_CPUs = os.cpu_count()


@click.group()
def cli():
    """Developper CLI: building (meson) sources in place, document, etc."""
    pass


############


@cli.command()
@click.option("--build-dir", default=str(BUILD_DIR), help="Relative path to the build directory")
@click.option(
    "-j",
    "--parallel",
    default=N_CPUs,
    show_default=True,
    help="Number of parallel jobs for building.",
)
def build(build_dir: str, parallel: int):
    """Build package using Meson build tool and install editable mode.

    \b
    ```python
    Examples:
    $ python dev.py build
    ```
    """
    # === setup build ===============================================
    cmd = ["meson", "setup", build_dir]
    if Path(build_dir).exists():
        cmd += ["--wipe"]
    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd, env=ENVS, cwd=BASE_DIR)
    if ret == 0:
        print("Meson build setup OK")
    else:
        print("Meson build setup failed!")
        sys.exit(1)

    # add __init__.py under cherab/
    # meson-cython compilation does not handle PEP420 even though cython is 3.0.
    # TODO: PEP420 handling
    temp_init = SRC_PATH / "__init__.py"
    with temp_init.open(mode="w") as file:
        file.write("# This file is automatically generated by dev.py build CLI.")

    # === build project =============================================
    cmd = ["meson", "compile", "-C", build_dir, "-j", str(parallel)]
    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd)

    if ret == 0:
        print("Build OK")
    else:
        print("Build failed!")
        # delete temporary __init__.py
        os.remove(temp_init)
        sys.exit(1)

    # === install .so files in source tree ==========================
    for so_path in BUILD_DIR.glob("**/*.so"):
        src = so_path.resolve()
        dst = BASE_DIR / so_path.relative_to(BUILD_DIR)
        shutil.copy(src, dst)
        print(f"copy {src} into {dst}")
    print("Install .so files in place.")

    # delete temporary __init__.py
    os.remove(temp_init)


@cli.command()
def install():
    """Install package as the editalbe mode.

    This command enables us to install the packages
    as an editable mode with the setuptools functionality.

    \b
    ```python
    Examples:
    $ python dev.py install
    ```
    """
    # install the package
    cmd = [sys.executable, "setup.py", "develop"]
    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd)

    if ret == 0:
        print("Successfully installed.")
    else:
        print("install failed!")
        sys.exit(1)


@cli.command()
def install_deps():
    """Install build dependencies using pip.

    Only pip install cannot compile cython files appropriately, so we excute this command before
    installing this package.
    """
    # Load requires from pyproject.toml
    pyproject = BASE_DIR / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError("pyproject.toml must be placed at the root directory.")

    with open(pyproject, "rb") as file:
        conf = tomllib.load(file)
    requires = conf["build-system"].get("requires")
    subprocess.run([sys.executable, "-m", "pip", "install"] + requires)


@cli.command()
@click.option("--data-dir", help="Relative path to the data stored directory", required=True)
@click.option(
    "--grid-filename", default="grid-360.txt", help="Grid data text file name", show_default=True
)
@click.option(
    "--cell-filename", default="CELL_GEO", help="Cell index data text file name", show_default=True
)
@click.option(
    "--store-dir",
    default="~/.cherab/lhd/",
    help="Relative directory path to store the data",
    show_default=True,
)
@click.option("--overwrite", is_flag=True, help="Overwrite the existing data", show_default=True)
def install_emc3_data(
    data_dir: str, grid_filename: str, cell_filename: str, store_dir: str, overwrite: bool
):
    """Install EMC3-EIRENE-related data including grids, indices, calculated data, etc. as a
    `emc3.hdf5` HDF5 file.

    This command should be excuted before using EMC3-related features if EMC3's HDF5 dataset does
    not been constructed. Note that it is only available after installing cherab-lhd.
    """
    try:
        from cherab.lhd.emc3.repository.install import (
            install_cell_indices,
            install_data,
            install_grids,
            install_physical_cell_indices,
        )
    except Exception as err:
        raise ImportError("cherab.lhd must be installed in advance.") from err

    # install grids
    install_grids(
        Path(data_dir) / grid_filename,
        hdf5_path=Path(store_dir).expanduser() / "emc3.hdf5",
        update=overwrite
    )
    install_physical_cell_indices(
        Path(data_dir) / cell_filename,
        hdf5_path=Path(store_dir).expanduser() / "emc3.hdf5",
        update=overwrite
    )
    install_cell_indices(
        hdf5_path= Path(store_dir).expanduser() / "emc3.hdf5",
        update=overwrite
    )
    install_data(Path(data_dir), hdf5_path=Path(store_dir).expanduser() / "emc3.hdf5")


############


@cli.command()
@click.argument("targets", default="html")
@click.option(
    "-j",
    "--parallel",
    default=N_CPUs,
    show_default=True,
    help="Number of parallel jobs for building.",
)
def doc(parallel: int, targets: str):
    """:wrench: Build documentation
    TARGETS: Sphinx build targets [default: 'html']
    """
    # move to docs/ and run command
    os.chdir("docs")

    builddir = DOC_ROOT / "build"
    srcdir = DOC_ROOT / "source"
    SPHINXBUILD = "sphinx-build"
    if targets == "html":
        cmd = [SPHINXBUILD, "-b", targets, f"-j{parallel}", str(srcdir), str(builddir / "html")]

    elif targets == "clean":
        cmd = ["rm", "-rf", str(builddir), "&&", "rm", "-rf", str(srcdir / "_api")]

    elif targets == "help":
        cmd = [SPHINXBUILD, "-M", targets, str(srcdir), str(builddir)]

    else:
        cmd = [SPHINXBUILD, "-M", targets, f"-j{parallel}", str(srcdir), str(builddir)]

    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd)

    if ret == 0:
        print("sphinx-build successfully done.")
    else:
        print("Sphinx-build has errors.")
        sys.exit(1)


############


@cli.command()
def format():
    """:art: Run ruff linting & formatting
    The default options are defined in pyproject.toml
    """
    cmd = ["ruff", "check", "--fix", str(SRC_PATH)]
    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd)

    if ret == 0:
        print("ruff formated")
    else:
        print("ruff formatting errors!")
        sys.exit(1)


@cli.command()
def cython_lint():
    """:art: Cython linter. Checking all .pyx files in the source directory.
    The default options are defined at the cython-lint table in pyproject.toml
    """
    # list of .pyx files
    pyx_files = [str(pyx_path) for pyx_path in SRC_PATH.glob("**/*.pyx")]

    cmd = ["cython-lint"] + pyx_files
    ret = subprocess.call(cmd)

    if ret == 0:
        print("cython-lint OK")
    else:
        print("cython-lint errors")
        sys.exit(1)


#######
def config(tool: str):
    """Load configure data from pyproject.toml for tool table."""
    pyproject = BASE_DIR / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError("pyproject.toml must be placed at the root directory.")

    with open(pyproject, "rb") as file:
        conf = tomllib.load(file)

    if not conf["tool"].get(tool):
        raise ValueError(f"{tool} config data does not exist.")

    return conf["tool"].get(tool)


if __name__ == "__main__":
    cli()
