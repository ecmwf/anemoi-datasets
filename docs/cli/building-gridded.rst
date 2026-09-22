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

   For gridded datasets, the per-group statistics are computed on the fly
   as each part is loaded and cached in the work directory.

#. **Finalise** the dataset. This merges the partial statistics, writes
   the metadata and attributes, and removes the temporary files:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr

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

Increment statistics (e.g. for 6h or 12h tendencies) are added with the
``*-additions`` commands, as described in :ref:`create-incremental`.

.. seealso::

   -  :ref:`layouts-gridded` — using a gridded dataset.
   -  :ref:`create-incremental` — the generic incremental build workflow.
