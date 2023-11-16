:orphan:

.. _installation:

============
Installation
============

.. note::

    Currently (07/07/2023), cherab-lhd requires some specific version of dependencies (`raysect`_
    and `cherab`_ ) because cherab-lhd handles some new features (like `TetraMesh` class)
    which have not merged to released branch yet.
    These dependencies has already listed in ``pyproject.toml`` in source directory,
    so those who are curious about it should look into the file.


Installing using pip
====================
Using ``pip`` command allows us to install cherab-lhd including dependencies.
For now, it is only available from `Cherab-lhd's GitHub repository`_.

.. prompt:: bash

    python -m pip install git+https://github.com/munechika-koyo/cherab_lhd




Installing for Developper
==========================
If you plan to make any modifications to do any development work on CHERAB-LHD,
and want to be able to edit the source code without having to run the setup script again
to have your changes take effect, you can install CHERAB-LHD on editable mode.

Manually downloading source
---------------------------
Before install the package, it is required to download the source code from github repository.
The source codes can be cloned from the GitHub reporepository with the command:

.. prompt:: bash

    git clone https://github.com/munechika-koyo/cherab_lhd

The repository will be cloned inside a new subdirectory named as ``cherab_lhd``.

Downloading data asset by Git LFS
---------------------------------
Data assets like device mesh files (``.rsm``, ``.STL``) are stored at the repository
by `Git LFS <https://git-lfs.github.com>`. After installing the Git LFS, the downloading data assets
can be accomplished by the following commands at the source root directory:

.. prompt:: bash

    git lfs install
    git lfs fetch

Building and Installing
-----------------------
Firstly, you need to install dependencies.
The easiest way is to create a conda development environment:

.. prompt:: bash

    conda env create -f environment.yaml  # `mamba` works too for this command
    conda activate cherab-lhd-dev

you need to build this package using the ``dev.py`` CLI:

.. prompt:: bash

    python dev.py build

This command enables us to compile cython codes with meson build-tool and put built shared object
(``.so``) files into the source tree.
This interface has some options, allowing you to perform all regular development-related tasks
(building, building docs, formatting codes, etc.).
Here we document a few of the most commonly used options; run ``python dev.py --help`` or ``--help``
on each of the subcommands for more details.

Additionally, to make a path to this package and register it as a `cherab` namespace package,
run the following command:

.. prompt:: bash

    python dev.py install

In this CLI, the ``setuptools`` shall install it into the ``**/site-packages/`` directory
as a namespace package with the develop (editable) mode.
