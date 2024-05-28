Introduction
============

When you install the `anemoi-datasets` package, this will also install command line tool
called ``anemoi-datasets`` which can be used to manage the zarr datasets.

The tool can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-datasets --help

The commands are:

.. toctree::
    :maxdepth: 1

    compare
    copy
    create
    inspect
    scan

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :nosubcommands:
