.. _nexus_record_command:

Nexus-record Command
====================

The ``nexus-record`` command prints a dataset's record for Anemoi Nexus: its
name (the directory name without ``.zarr``), its ``uuid`` and, as
``metadata``, its zarr attributes with the statistics and the ``data``
array's shape, dtype and chunks. ``nexus-client`` sends it as is, and never
reads the zarr itself.

.. code:: console

   $ anemoi-datasets nexus-record my-dataset.zarr @attributes.yaml -o record.json
   $ nexus-client create datasets my-dataset --file record.json

The record attributes Nexus needs to file the asset are given as options
(``--owner``, ``--project``/``--projects``, ``--license``/``--licenses``,
``--name``) or in an ``@FILE`` (JSON or YAML), in which ``project``/``projects``
and ``license``/``licenses`` may be singular or plural; the options win over
the file:

.. code:: yaml

   # attributes.yaml
   owner: alice
   project: MLP
   licenses: [CC-BY-4.0]

The attributes are handled by :mod:`anemoi.utils.nexus`, shared with
``anemoi-inference nexus-record``.

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: nexus-record
