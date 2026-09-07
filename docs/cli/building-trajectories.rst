.. _building-trajectories-cli:

#######################################
 Building a trajectory dataset (CLI)
#######################################

This page walks through building a :ref:`trajectory
<layouts-trajectories>` dataset from a recipe using the
``anemoi-datasets`` command line tool. For the recipe syntax itself, see
:ref:`building-introduction`.

A trajectory dataset stores forecast fields indexed by a *base date* (the
model-run time) and a *forecast step*, rather than a single validity
time. The on-disk array is 5-D ``(base_dates, variables, ensembles,
steps, cells)``.

**********
 Recipe
**********

Set ``output.layout`` to ``trajectories`` and replace the usual
``dates:`` block with two blocks, ``base_dates:`` and ``steps:``. The set
of samples written on disk is the Cartesian product of the base dates and
the steps.

.. code:: yaml

   base_dates:
     start: 2021-01-01 00:00:00
     end:   2021-01-02 00:00:00
     frequency: 12h

   steps:
     start: 6
     end: 30
     frequency: 6h

   input:
     mars:
       type: fc
       # ... source definition ...

   output:
     layout: trajectories

.. note::

   ``base_dates:`` and ``steps:`` are **required** for
   ``layout: trajectories`` and ``dates:`` is rejected. Conversely, for
   any other layout ``dates:`` is required. See :ref:`layouts-trajectories`
   for the full set of rules.

*******************
 One-shot creation
*******************

The simplest way to build the dataset is the :ref:`create command
<create_command>`, which runs every step in a single process:

.. code:: bash

   anemoi-datasets create dataset.yaml dataset.zarr --overwrite

**************************
 Incremental / parallel
**************************

For large datasets, build the dataset step by step so the loading can be
split across processes, terminals or SLURM jobs.

#. **Initialise** the (empty) dataset:

   .. code:: bash

      anemoi-datasets init dataset.yaml dataset.zarr --overwrite

#. **Load** the data in parts. Parts are numbered ``1/N`` … ``N/N``
   (1-based); each part loads a subset of the base dates and can be run in
   any order and in parallel:

   .. code:: bash

      anemoi-datasets load dataset.zarr --parts 1/10
      anemoi-datasets load dataset.zarr --parts 2/10
      # ... up to ...
      anemoi-datasets load dataset.zarr --parts 10/10

#. **Finalise** the dataset (merge statistics, write metadata and
   attributes, clean up temporary files):

   .. code:: bash

      anemoi-datasets finalise dataset.zarr

#. **Patch** the metadata:

   .. code:: bash

      anemoi-datasets patch dataset.zarr

You can follow the progress at any time and clean up leftover temporary
files with:

.. code:: bash

   anemoi-datasets inspect dataset.zarr
   anemoi-datasets cleanup dataset.zarr

.. seealso::

   -  :ref:`layouts-trajectories` — using a trajectory dataset and the
      full recipe rules.
   -  :ref:`create-incremental` — the generic incremental build workflow.
