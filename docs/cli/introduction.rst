##############
 Introduction
##############

When you install the `anemoi-datasets` package, this will also install
command line tool called ``anamois-datasets`` this can be used to manage
the zarr datasets.

The tools can provide help with the ``--help`` options:

.. code:: bash

   % anamoi-datasets --help

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
