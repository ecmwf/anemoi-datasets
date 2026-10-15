.. _building-gridded-cli:

##################################
 Building a gridded dataset (CLI)
##################################

This page walks through building a :ref:`gridded <layouts-gridded>`
dataset from a recipe using the ``anemoi-datasets`` command line tool.
For the recipe syntax itself, see :ref:`building-introduction`.

A gridded dataset stores model fields on a (possibly unstructured)
spatial grid that is regular in time. On disk it is a 4-D array
``(dates, variables, ensembles, grid_points)``.

**********
 Recipe
**********

Set ``output.layout`` to ``gridded`` in the recipe. This is the default,
so it may be omitted:

.. code:: yaml

   dates:
     start: 2020-01-01 00:00:00
     end: 2020-12-31 18:00:00
     frequency: 6h

   input:
     mars:
       # ... source definition ...

   output:
     layout: gridded

*******************
 One-shot creation
*******************

The simplest way to build the dataset is the :ref:`create command
<create_command>`, which runs every step in a single process:

.. code:: bash

   anemoi-datasets create dataset.yaml dataset.zarr --overwrite

Before running a full build, you can generate a reduced recipe with the
:ref:`create-test-recipe command <create_test_recipe_command>` and build
that first to check the configuration.

**************************
 Incremental / parallel
**************************

For large datasets, build the dataset step by step so the loading can be
split across processes, terminals or SLURM jobs. This mirrors the general
:ref:`incremental build <create-incremental>` workflow.

#. **Initialise** the (empty) dataset. The recipe is copied into the
   store, so it is no longer needed by the following steps:

   .. code:: bash

      anemoi-datasets init dataset.yaml dataset.zarr --overwrite

#. **Load** the data in parts. Parts are numbered ``1/N`` … ``N/N``
   (1-based) and can be run in any order and in parallel; ``zarr``
   handles the concurrent writes:

   .. code:: bash

      anemoi-datasets load dataset.zarr --parts 1/20
      anemoi-datasets load dataset.zarr --parts 2/20
      # ... up to ...
      anemoi-datasets load dataset.zarr --parts 20/20

   The loaders only write the data; the statistics are computed by the
   ``finalise`` step.

#. **Finalise** the dataset. This computes the statistics (including the
   tendency statistics, see ``statistics.tendencies`` in the recipe),
   writes the metadata and attributes, and removes the temporary files:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr

   A single ``finalise`` computes all the statistics in one process,
   reading the whole dataset. The finalise can also be run in parallel,
   in three stages:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr --prepare
      anemoi-datasets finalise dataset.zarr --load --parts 1/10
      # ... up to ...
      anemoi-datasets finalise dataset.zarr --load --parts 10/10
      anemoi-datasets finalise dataset.zarr --tidy

   ``--prepare`` checks that every part has been loaded and removes the
   partial statistics left by a previous run. Each ``--load`` computes
   the statistics of a contiguous range of dates (whole chunks); it also
   reads the few dates that precede its range, so that the tendencies
   that straddle two ranges are computed. The ``--load`` parts can run in
   any order and in parallel, and a part that has completed is skipped if
   it is run again. ``--tidy`` merges the partial statistics, writes them
   with the metadata, and removes the temporary files.

#. **Patch** the metadata (this removes the reference to the recipe file
   used at ``init`` time):

   .. code:: bash

      anemoi-datasets patch dataset.zarr

You can follow the progress at any time with:

.. code:: bash

   anemoi-datasets inspect dataset.zarr

If temporary files are left behind, remove them with:

.. code:: bash

   anemoi-datasets cleanup dataset.zarr

***********************
 Additional statistics
***********************

Increment statistics (e.g. for 6h or 12h tendencies) are computed by the
``finalise`` step, for the deltas listed in ``statistics.tendencies``.
The ``*-additions`` commands are deprecated and do nothing.

.. seealso::

   -  :ref:`layouts-gridded` — using a gridded dataset.
   -  :ref:`create-incremental` — the generic incremental build workflow.
