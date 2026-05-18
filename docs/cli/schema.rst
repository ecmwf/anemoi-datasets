.. _schema_command:

Schema Command
==============

The `schema` command exports the JSON schema of a recipe. This can be useful for validating
recipe files or for integrating with tools that support JSON schema.

.. code:: console

   $ anemoi-datasets schema

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: schema
