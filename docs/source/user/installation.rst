:orphan:

.. _installation:

============
Installation
============


Installing using pip
====================
Using ``pip`` command allows us to install cherab-lhd including dependencies.
For now, it is only available from `Cherab-lhd's GitHub repository`_.

.. prompt:: bash

    pip install git+https://github.com/munechika-koyo/cherab_lhd




Installing for Development
==========================
If you plan to make any modifications to / do any development work on CHERAB-LHD,
and want to be able to edit the source code without having to run the setup script again
to have your changes take effect, you can install CHERAB-LHD on eidtable mode.

manually downloaded source
--------------------------
The source codes can be cloned from the GitHub reporepository with the command:
Before install the cherab-iter, it is required to download the source code from github repository.

.. prompt:: bash

    git clone https://github.com/munechika-koyo/cherab_lhd

The repository will be cloned inside a new subdirectory named as ``cherab_lhd``.

Building and Installing
-----------------------
In the ``cherab_lhd`` directory, run

.. prompt:: bash

    python -m pip install -e .

Where ``-e`` option enables us to build the source code just where you excute it
and generate the python path into the ``site-packages`` directory.
