.. _inspect_command:

Inspect Command
===============


Anemoi datasets are stored in a zarr format and can be located on a local file system or on a remote server.
The `inspect` command is used to inspect the contents of a dataset.
This command will output the metadata of the dataset, including the variables, dimensions, and attributes.

.. code:: console

   $ anemoi-datasets inspect dataset.zarr


which will output something like the following. The output should be self-explanatory.

.. literalinclude:: ../datasets/yaml/building1.txt
   :language: console

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: inspect
