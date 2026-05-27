.. _create_test_recipe_command:

Create Test Recipe Command
==========================

Use this command to create a reduced test recipe from a full anemoi-datasets recipe YAML file.
The syntax of the recipe file is described in :ref:`building-introduction`.

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: create-test-recipe
