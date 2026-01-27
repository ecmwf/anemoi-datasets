.. _recipe_command:

Recipe Command
==============


Anemoi datasets are stored in a zarr format and can be located on a local file system or on a remote server.
The `inspect` command is used to inspect the contents of a dataset.
This command will output the metadata of the dataset, including the variables, dimensions, and attributes.

.. code:: console

   $ anemoi-datasets recipe [options] recipe.yaml


.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: recipe
