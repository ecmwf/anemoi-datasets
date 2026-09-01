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
huge number of fragments, it can be split into three stages (plus an
optional reporting step) so the ``zarr`` array is populated incrementally
and, if you wish, in parallel from several processes. Running ``finalise``
with no stage flag (as above) runs the stages in one process and is fully
equivalent.

#. **Prepare** — deduplicate and de-overlap the fragments, compute the
   final shape, create the (empty) ``zarr`` array and write a manifest
   (the list of fragments and their position in the final array) into the
   work directory. Run this **once**:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr --prepare

#. **Rows per chunk** — choose the ``rows_per_chunk`` that minimises read time
   when iterating the dataset with a fixed time window. Run this **once**,
   after ``--prepare`` and before ``--load``:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr --rows-per-chunk

   By default (``output.auto_rows_per_chunk``, ``1d``) this computes the
   optimum for that window, stores it as ``output.rows_per_chunk`` and
   **re-chunks the (still empty) data array** to match. Add ``--print`` to
   instead just print the optimum for every window and change nothing:

   .. code:: bash

      anemoi-datasets finalise dataset.zarr --rows-per-chunk --print
      # e.g. {'1h': 1238797, '3h': 2406724, '6h': 3087352, '12h': 4675762, '24h': 5520252}

   See :ref:`tabular-rows-per-chunk` below for how the optimum is computed and
   how to control it. If you set ``output.rows_per_chunk`` yourself in the
   recipe, this step is skipped and your value is kept.

#. **Load** — write a part of the data into the ``zarr`` array. Parts are
   split along **whole zarr chunks** (using the 1-based ``--parts i/n``
   convention of the ``load`` command), so no chunk is ever written by two
   ``--load`` calls and the parts can be run in any order and fully in
   parallel. Each part also produces partial statistics and a partial date
   index:

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

   The ``--prepare`` / ``--rows-per-chunk`` / ``--load`` / ``--tidy`` flags
   are mutually exclusive. As with the ``load`` command, put the dataset path
   before ``--parts`` on the command line.

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

.. note::

   Because parts are aligned to whole zarr chunks, each ``--load`` call
   owns a disjoint set of chunks — no chunk is written by more than one
   call — so the parts are safe to run concurrently. A fragment that spans
   a part boundary is *read* by both neighbouring parts, but each writes
   only the rows in its own chunks. A date whose observations straddle a
   boundary is stitched back into a single index entry during ``--tidy``.

.. _tabular-rows-per-chunk:

*******************************
 Choosing the rows per chunk
*******************************

``zarr`` re-reads and decompresses a **whole chunk** on every access. When
you iterate a tabular dataset with a fixed time window — reading
``[d, d+w)``, then ``[d+w, d+2w)``, and so on — each read pulls, in full,
every chunk its rows overlap. Chunks that are too small make each read touch
many chunks (and pay the filesystem's per-read latency many times); chunks
that are too large re-read data the next window already loaded and can exceed
the size the filesystem serves efficiently. The ideal chunk holds roughly one
window's worth of rows — but because the row density varies over the dataset's
time range (a new observing system may start sparse and ramp up to full
production), no single value lines up with every window.

The ``--rows-per-chunk`` stage estimates the best value for you by modelling
the actual **read time** of a full sweep. For each window size it minimises

.. code:: text

   time(C) = read_time(C) * sum over windows of (chunks touched by the window)

where ``C`` is the chunk size in rows. A chunk holds ``C * row_bytes`` raw
bytes, which compress to ``C * row_bytes * compression_ratio`` on disk, and

.. code:: text

   read_time(C) = fs_latency + on_disk_bytes / effective_bandwidth

The effective bandwidth is at its peak for reads inside the filesystem's sweet
spot (by default **64–512 MB** on disk) and falls off for smaller reads
(latency-dominated) and larger ones (throughput rolls off). This keeps the
chosen chunk both well-sized for the filesystem and close to one window.

The row density comes from an **exact per-timestamp histogram** collected
during ``load`` (the dates are in memory as the fragments are written, so it is
free), falling back to an estimate from the fragment manifest if no histogram
is present. Each window size is treated independently. For each, the stage
reports the **time-optimal** ``rows_per_chunk`` and also logs the
**band-clamped** alternative — "one window per chunk, clamped into the
filesystem sweet spot" — which gives the fewest reads while each read still
lands in the fast band. The mean/min/max rows per window and the resulting
on-disk chunk size are logged too.

What the stage *does* with the result depends on the mode:

-  **Default (apply)** — when ``output.auto_rows_per_chunk`` is set (default
   ``1d``) and ``output.rows_per_chunk`` is still unset, the optimum for that
   one window is stored as ``output.rows_per_chunk`` (in the dataset metadata)
   and the empty ``data`` array is **re-chunked** to match, ready for
   ``--load``.
-  **Print only** (``--print``) — every window in ``chunk_windows`` is
   evaluated and printed; nothing is changed.
-  **User-set** — if you set ``output.rows_per_chunk`` yourself, the automatic
   step is skipped and your value is kept (``--prepare`` already chunked the
   array with it). Running ``--rows-per-chunk`` explicitly in this case (or
   with ``auto_rows_per_chunk`` disabled) is an error; use ``--print``.

The behaviour is controlled from the recipe's ``output`` section:

-  ``auto_rows_per_chunk`` — the single window used to choose ``rows_per_chunk``
   automatically (default ``1d``; set to null to require an explicit
   ``rows_per_chunk``).
-  ``rows_per_chunk`` — set it explicitly to skip the automatic choice.

The read-time model is tuned in the recipe's ``build`` section:

-  ``chunk_windows`` — the window sizes evaluated by ``--print`` (default
   ``1h, 3h, 6h, 12h, 24h``).
-  ``chunk_alignment_offset`` — shift the window boundaries off the whole UTC
   hour, e.g. ``30min`` or ``3h`` for a different time zone (default none, so
   windows end on a whole UTC hour).
-  ``chunk_compression_ratio`` — on-disk bytes ÷ raw bytes (default ``0.5``).
-  ``fs_read_min_bytes`` / ``fs_read_max_bytes`` — the filesystem read sweet
   spot in bytes (defaults 64 MB / 512 MB).
-  ``fs_read_latency_seconds`` — fixed per-read cost (default ``0.005``).
-  ``fs_read_bandwidth_bytes_per_s`` — peak streaming bandwidth (default
   ``2e9``).

You can follow the progress at any time and clean up leftover temporary
files with:

.. code:: bash

   anemoi-datasets inspect dataset.zarr
   anemoi-datasets cleanup dataset.zarr

.. seealso::

   -  :ref:`layouts-tabular` — using a tabular dataset (windows,
      samples, auxiliary information).
   -  :ref:`create-incremental` — the generic incremental build workflow.
