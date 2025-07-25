.. _validate_command:

Validate Command
============

Use this command to validate a zarr dataset, or a class that implements the
:class:`anemoi.datasets.Dataset` interface.


This command has two modes of operation:

1. Validate a local zarr dataset by specifying its path.
2. Validate a dataset class by providing its fully qualified class name.

Example usage
-------------

To validate a local zarr dataset:

.. code-block:: bash

    anemoi-datasets validate /path/to/dataset.zarr

To validate a dataset class:

.. code-block:: bash

    anemoi-datasets validate --callable my_package.MyDatasetClass  some-args-relevant-to-the-class


In the first case, the command will check the compatibility of the zarr dataset using anemoi-datasets own class, in the second case, it will use the class provided by the user.

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: validate
