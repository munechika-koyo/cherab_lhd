:orphan:

.. _installation:

============
Installation
============

.. note::

    Currently (02/14/2025), cherab-lhd requires some specific version of dependencies (`raysect`_
    and `cherab`_ ) because cherab-lhd handles some new features (like `TetraMeshData` class)
    which have not merged to released branch yet.
    These dependencies has already listed in ``pyproject.toml`` in source directory,
    so those who are curious about it should look into the file.


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


For Developers
==============
If you want to install from source in order to contribute to develop `cherab-lhd`,
`Pixi`_ is required for several development tasks, such as building the documentation and running the tests.
Please install it by following the `Pixi Installation Guide<https://pixi.sh/latest#installation>` in advance.

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

    >>> import cherab.lhd
    >>> # Begin interactive work with cherab-lhd

Additionally, useful commands for development are shown in the :ref:`contribution` section.
