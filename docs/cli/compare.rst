.. _compare_command:

Compare Command
===============

Use this command to compare two datasets.

The command will run a quick comparison of the two datasets and output a summary of the differences.

.. warning::

    This command will not compare the data in the datasets, only some of the metadata.
    Subsequent versions of this command may include more detailed comparisons.


.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: compare
