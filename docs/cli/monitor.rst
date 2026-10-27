.. _monitor_command:

Monitor Command
===============

The ``monitor`` command plots a dataset's build diagnostics and monthly
maps into ``<dataset>/_monitor/``:

- ``timeseries.png``: per day and group (e.g. satellite), the cells
  filled and the mean ± std of each variable;
- ``qc.png``: per day and group, the rows read, the fraction each QC
  step removes, and the fraction of cells filled;
- ``maps_YYYY-MM.png``: for one month, per grid cell, the fraction of
  steps filled, the fraction of rows rejected by QC, and the mean, std
  and one mid-month step of each map variable.

.. code:: console

   $ anemoi-datasets monitor dataset.zarr
   $ anemoi-datasets monitor dataset.zarr --month 2020-01 --variables obsvalue_rawbt_921 n_obs

The diagnostics are recorded by sources such as :ref:`grid_observations
<grid-observations-source>` while the dataset is built, and merged into
``<dataset>/_monitor/data/`` by ``finalise``. Plots whose inputs are
missing are skipped. By default the maps show the last month, and the
recipe's ``maps``. Maps read the dataset itself, so they work for any
gridded dataset with ``--variables``.

Needs ``matplotlib`` (the ``observations`` extra).

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: monitor
