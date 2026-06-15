.. _extract_command:

Extract Command
===============

The ``extract`` command extracts constant fields and/or climatologies from a
dataset and writes them to a NetCDF file following CF conventions.

This is useful for producing ancillary files needed during inference, such as
orography (a constant) or monthly mean sea-surface temperature (a climatology).

The ``DATASET`` argument can be:

- A dataset name or path (resolved by ``open_dataset``).
- A YAML or JSON file (ending in ``.yaml`` or ``.json``) whose content is
  loaded and passed to ``open_dataset``.

Usage examples
--------------

Extract a constant field:

.. code:: console

   $ anemoi-datasets extract --constant z dataset-name output.nc

Extract multiple constants:

.. code:: console

   $ anemoi-datasets extract --constant z --constant lsm dataset-name output.nc

Extract climatologies (one value per month):

.. code:: console

   $ anemoi-datasets extract --climatology sst --climatology ci dataset-name output.nc

Mix constants and climatologies:

.. code:: console

   $ anemoi-datasets extract --constant z --climatology sst dataset-name output.nc

Use a YAML file to specify the dataset:

.. code:: console

   $ anemoi-datasets extract --constant z dataset.yaml output.nc

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: extract
