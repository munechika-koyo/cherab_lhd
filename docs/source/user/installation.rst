:orphan:

.. _installation:

============
Installation
============

For Users
=========
`cherab-lhd` can be installed by many package managers.
Explore the various methods below to install `cherab-lhd` using your preferred package manager.

.. tab-set::

    .. tab-item:: pip

        ::

            pip install cherab-lhd

    .. tab-item:: uv

        ::

            uv add cherab-lhd


    .. tab-item:: conda

        ::

            conda install -c conda-forge cherab-lhd

    .. tab-item:: pixi

        ::

            pixi add cherab-lhd

.. attention::

    `cherab-lhd` package does not provide any LHD-related data (grid, machine mesh, measurement info, etc.)
    because of the large size.
    When functions requiring data are executed, `pooch` is used to automatically download the data using sftp.
    However, since this involves credential information, authentication is required.
    Currently, the data is not publicly available, so users who need access to the data should contact the administrator.
    See also :ref:`Data Handling <data>`.


For Developers
==============
If you want to install from source in order to contribute to develop `cherab-lhd`,
`Pixi`_ is required for several development tasks, such as building the documentation and running the tests.
Please install it by following the `Pixi Installation Guide <https://pixi.sh/latest#installation>`_ in advance.

Afterwards, you can install `cherab-lhd` by following three steps:

1. Clone the `cherab-lhd` repository::

    git clone https://github.com/munechika-koyo/cherab_lhd

2. Enter the repository directory:

    cd cherab_lhd

3. Install the package:

    pixi install

`pixi` install required packages into the isolated environment, so you can develop `cherab-lhd` without worrying about the dependencies.
To use cherab-lhd in interactive mode, launch the Python interpreter by executing::

    pixi run python

Once the interpreter is running, you can import and use cherab-lhd, for example::

    >>> from cherab.lhd import __version__
    >>> print(__version__)

Additionally, useful commands for development are shown in the :ref:`contribution` section.
