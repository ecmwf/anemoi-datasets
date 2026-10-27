.. _create-gridded-observations:

#########################################
 Create a gridded observation dataset
#########################################

This how-to builds a gridded dataset from satellite observations. In
the output, each date *T* holds, per grid cell, one observation from
*T*'s time window, plus the observation's timing and geometry and
forcings at the grid point. It uses the :ref:`grid_observations
<grid-observations-source>` source. The output is an ordinary gridded
dataset: ``data[date, variable, 0, cell]``, with NaN where a window has
no observation, read like any other.

**********
 A recipe
**********

.. literalinclude:: ../../building/sources/yaml/grid-observations.yaml
   :language: yaml

The parts of the recipe:

``dates``
   One date per window end *T*, every ``frequency``.

``build.group_by``
   The unit of work: one group is loaded at a time, and each
   ``load --parts`` process gets a block of groups. ``daily`` reads each
   day's observations once for its four 6-hour windows.

``window``
   Which observation times *T* holds, relative to *T*. ``(-350m,+10m]``
   is ``T - 5h50 < time <= T + 10min``. The previous day's tail is read
   automatically.

``select``
   What fills a cell: the nearest pixel of the latest overpass, or the
   latest pixel, or the overpass mean.

``qc``
   Which rows may be selected (``flags``, ``require_complete``) and which
   values are masked (``range``, ``known_bad``).

``variables``
   What is stored: channels and other columns, derived timing and
   geometry, forcings, spreads, and output names.

``input``
   Where the rows come from: here per-day parquet files. Any tabular
   source works, or a ``pipe`` of one and tabular filters.

*************************
 Building it in parallel
*************************

A dataset is built in three steps. ``load`` can run as many processes
at once as you like, each on its own block of groups:

.. code:: console

   $ anemoi-datasets init recipe.yaml dataset.zarr
   $ anemoi-datasets load dataset.zarr --parts 1/240 &   # ... up to 240/240
   $ anemoi-datasets finalise dataset.zarr

Every group is written to its own chunks and marked done, so parts run
in any order on any nodes. A failed part can simply be run again: it
skips the groups already done. On Slurm, run several parts per job
(e.g. 16 processes on 16 CPUs) rather than one job per part, because
schedulers often limit jobs per user.

Size the memory per process from one heavy day:

.. code:: console

   $ /usr/bin/time -f "%M KB" anemoi-datasets load test.zarr

Per process, a group needs roughly:

-  its input rows (independent of the grid);
-  its output: variables × windows in the group × grid points × 4
   bytes, roughly twice over;
-  one date chunk for the write buffer.

For a day of IASI (3.4 M rows) with 44 variables, this was 2.5 GB on
o96 and 3.2 GB on N320. On high-resolution grids (O1280: 9.6 GB per day),
use ``group_by: 6h`` to load one window per group (6.4 GB).

**********
 Checking
**********

.. code:: console

   $ anemoi-datasets inspect dataset.zarr
   $ anemoi-datasets monitor dataset.zarr

See :ref:`monitor_command`. Check the QC fractions, the cells filled per
group (satellite changes show up there), and the maps of a few channels
and of ``time_delta`` and ``n_obs``. The swath pattern should match the
instrument.

***************************
 Real time from ODB files
***************************

The same recipe builds real-time inference input from per-cycle ODB
files: replace the inner ``parquet`` by an :ref:`odb <odb-source>`
source whose path is expanded with the cycle date, and set
``milliseconds_column_name`` if the files carry one. ODB times are
otherwise whole seconds, which moves rows at window edges and changes
which pixel wins a tie. A ``rename_tabular`` filter in a ``pipe`` can map
ODB column names to the ones the recipe uses.

***********************************
 Adding a new observation type
***********************************

#. Start from the recipe of the closest existing instrument: microwave
   sounders and infrared sounders differ mainly in channels, scan
   geometry and QC.

#. Pick the ``channels`` and the extra ``columns`` you need. Their names
   are the input's, renamed with ``rename``.

#. Set ``scan`` from the instrument's field-of-view numbering
   (``n_positions``, ``group_size``, the side position 0 views). Check
   the sign of ``scan_pos`` against the geometry.

#. Measure the QC rather than guess it: build one month without
   ``range``, look at ``qc.png`` and the statistics, then set the ranges
   and flags.

#. Choose ``select``. ``latest_pass`` + ``nearest`` keeps real spectra;
   ``mean`` (with ``mean_of: radiance`` for brightness temperatures)
   superobs an overpass.

#. Build one heavy month, size the memory, then build the full period
   in parallel.
