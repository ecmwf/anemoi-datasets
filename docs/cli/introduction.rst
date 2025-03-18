.. _cli-introduction:

##################
Command line tool
##################

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


.. Create Command
.. --------------

.. .. toctree::
..     :maxdepth: 1
..     create


.. .. argparse::
..     :module: anemoi.datasets.__main__
..     :func: create_parser
..     :prog: anemoi-datasets
..     :path: create

.. Inspect Command
.. ----------------

.. .. toctree::
..     :maxdepth: 1
..     inspect

.. .. argparse::
..     :module: anemoi.datasets.__main__
..     :func: inspect_parser
..     :prog: anemoi-datasets
..     :path: inspect

.. Compare Command
.. ---------------

.. .. toctree::
..     :maxdepth: 1
..     compare

.. .. argparse::
..     :module: anemoi.datasets.__main__
..     :func: compare_parser
..     :prog: anemoi-datasets
..     :path: compare

.. Copy Command
.. ---------------

.. .. toctree::
..     :maxdepth: 1
..     copy

.. .. argparse::
..     :module: anemoi.datasets.__main__
..     :func: copy_parser
..     :prog: anemoi-datasets
..     :path: copy