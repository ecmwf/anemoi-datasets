.. _create_command:

Create Command
==============

Use this command to create a dataset from a recipe file.
The syntax of the recipe file is described in :doc:`building datasets <../datasets/building/introduction>`.

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: create
