.. _create_command:

Create Command
==============

Use this command to create a dataset from a recipe file.
The syntax of the recipe file is described in :ref:`building-introduction`.

Before running a full dataset build, you can generate a reduced recipe with
:ref:`Create Test Recipe Command <create_test_recipe_command>` and run
``anemoi-datasets create`` on that test recipe first.

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: create
