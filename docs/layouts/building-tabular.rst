.. _building-tabular-cli:

##################################
 Building a tabular dataset (CLI)
##################################

This page walks through building a :ref:`tabular <layouts-tabular>`
dataset from a recipe using the ``anemoi-datasets`` command line tool.
For the recipe syntax itself, see :ref:`building-introduction`.

A tabular dataset stores observations that are unstructured both in time
and space. During loading, the observations are written as many small
*fragment* files in a work directory; the ``finalise`` step then
deduplicates them, packs them into a single, time-ordered ``zarr`` array
and builds a date index. This makes the tabular ``finalise`` step heavier
than for other layouts, which is why it can itself be split across
processes (see :ref:`tabular-staged-finalise` below).

**********
 Recipe
**********

Set ``output.layout`` to ``tabular`` in the recipe:

.. literalinclude:: code/tabular.yaml

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

For large datasets, build the dataset step by step so the work can be
split across processes, terminals or SLURM jobs.

#. **Initialise** the (empty) dataset:

   .. code:: bash

      anemoi-datasets init dataset.yaml dataset.zarr --overwrite

#. **Load** the observations in parts. Parts are numbered ``1/N`` …
   ``N/N`` (1-based) and can be run in any order and in parallel. Each
   part writes its observations as fragment files in the work directory:

   .. code:: bash

      anemoi-datasets load dataset.zarr --parts 1/20
      anemoi-datasets load dataset.zarr --parts 2/20
      # ... up to ...
      anemoi-datasets load dataset.zarr --parts 20/20

#. **Finalise** the dataset (deduplicate the fragments, pack them into
   the ``zarr`` array, compute statistics, build the date index):

   .. code:: bash

      anemoi-datasets finalise dataset.zarr

#. **Patch** the metadata:

   .. code:: bash

      anemoi-datasets patch dataset.zarr

.. _tabular-staged-finalise:

*******************************
 Splitting the finalise step
*******************************

Because the tabular ``finalise`` deduplicates and packs a potentially
huge number of fragments, it can be split into three stages so the
``zarr`` array is populated incrementally and, if you wish, in parallel
from several processes. Running ``finalise`` with no stage flag (as
above) runs all three stages in one process and is fully equivalent.

#. **Prepare** — deduplicate and de-overlap the fragments, compute the
   final shape, create the (empty) ``zarr`` array and write a manifest
   (the list of fragments and their position in the final array) into the
   work directory. Run this **once**:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr --prepare

#. **Load** — write a part of the fragments into the ``zarr`` array,
   using the manifest to know where each fragment goes. Parts use the
   same 1-based ``--parts i/n`` convention as the ``load`` command and can
   be run in any order and in parallel. Each part also produces partial
   statistics and a partial date index:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr --load --parts 1/5
      anemoi-datasets finalise dataset.zarr --load --parts 2/5
      # ... up to ...
      anemoi-datasets finalise dataset.zarr --load --parts 5/5

#. **Tidy** — merge the partial statistics, concatenate the partial date
   indexes, build the final index, write the attributes and delete the
   temporary files (the manifest is only removed here, in this last
   step). Run this **once**, after all parts are loaded:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr --tidy

.. note::

   The ``--prepare`` / ``--load`` / ``--tidy`` flags are mutually
   exclusive. As with the ``load`` command, put the dataset path before
   ``--parts`` on the command line.

.. note::

   Every stage is **re-runnable and crash-safe**, which matters when the
   commands are driven by a scheduler such as SLURM and may be killed at
   any time. A small journal of marker files in the work directory records
   what has been committed:

   -  ``--prepare`` never rewrites a fragment in place: de-overlapping
      writes new files and deletes each pair's inputs only once the outputs
      are durable, reclaiming disk as it goes so peak usage stays bounded
      (a fragment may be de-overlapped repeatedly, so keeping every
      generation could otherwise blow up disk use on a multi-terabyte
      build). A journal (``fsync``-ed before each delete) lets an
      interrupted ``--prepare`` reclaim any stragglers and resume without
      losing data. Once it has completed, re-running it is a no-op.
   -  A ``--load`` part is only marked done once its data, statistics and
      date ranges are all written; its fragment files are deleted only
      after that. Re-running an interrupted part recomputes it; re-running
      a finished part just removes any leftover fragment files.
   -  ``--tidy`` marks the dataset finalised in the store before removing
      any temporary files, so re-running it (or any other stage) after the
      dataset is complete — even after the work directory has been cleaned
      up — is a no-op.

.. warning::

   The staged ``--load`` writes directly to the ``zarr`` array and relies
   on ``zarr`` to handle partial chunks. If two ``--load`` processes write
   to the *same* chunk at a part boundary at the same time, one write can
   overwrite the other. Running the parts sequentially (or one process per
   disjoint chunk range) avoids this.

You can follow the progress at any time and clean up leftover temporary
files with:

.. code:: bash

   anemoi-datasets inspect dataset.zarr
   anemoi-datasets cleanup dataset.zarr

.. seealso::

   -  :ref:`layouts-tabular` — using a tabular dataset (windows,
      samples, auxiliary information).
   -  :ref:`create-incremental` — the generic incremental build workflow.
