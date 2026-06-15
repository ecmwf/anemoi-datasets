.. _layouts-trajectories:

##############
 Trajectories
##############

.. note::

   This page describes what is specific to the trajectories layout. For
   more general information on creating and using datasets, see
   :ref:`using-introduction` and :ref:`building-introduction`
   respectively.

A *trajectory* dataset stores forecast fields indexed by a *base date*
(the model-run time) and a *forecast step*, rather than a single validity
time. The on-disk array is 5-D ``(base_dates, variables, ensembles,
steps, cells)``.

.. _trajectories-creating:

**********
 Creating
**********

To create a trajectory dataset, set ``output.layout`` to ``trajectories``
and replace the usual ``dates:`` block by two blocks, ``base_dates:``
and ``steps:``.

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
       class: od
       expver: "0001"
       grid: 20./20.
       param: [q, t]
       levtype: pl
       level: [50]
       stream: oper

   output:
     layout: trajectories

Rules enforced by the recipe validator:

-  ``base_dates:`` and ``steps:`` are **required** for
   ``layout: trajectories``. ``dates:`` is rejected.
-  For any other layout, ``dates:`` is required and ``base_dates:`` /
   ``steps:`` are rejected.

``base_dates`` accepts the same ``start`` / ``end`` / ``frequency`` /
``missing`` keys as the regular ``dates`` block. ``steps`` accepts
``start``, ``end`` and ``frequency`` as time-delta specifications
(``6`` meaning 6 hours, or ``"6h"``, ``"1d"``, …). The set of samples
materialised on disk is the Cartesian product of basetimes and steps.

Which sources can feed a trajectory recipe
==========================================

The trajectory pipeline passes ``(valid_time, basetime)`` pairs to each
source. Only sources that know how to handle forecast-indexed requests
can be used at the leaves of a trajectory recipe. Today this means:

-  :ref:`mars <sources-mars>` — full forecast and forecast-accumulation
   support.
-  :ref:`accumulate <sources-accumulate>` — forecast accumulations via
   the ``accumulation: from-zero | from-previous-step`` flag; see below.

``grib-index``, ``fdb``, ``hindcasts`` and ``recentre`` are not
trajectory-aware in this release. ``from-trajectories:`` is the reverse
bridge: it lets a regular :ref:`gridded <layouts-gridded>` recipe pull
fields from a forecast archive; see :ref:`sources-from-trajectories`.

Forecast accumulations
======================

Inside a trajectory recipe, ``accumulate:`` produces per-step
accumulation fields anchored on the caller-imposed basetime. The
``accumulation`` flag is **required** and picks the archive convention:

-  ``from-zero`` — the archive stores accumulations from the basetime
   (``a(0, step)``). The window ``[bt+sA, bt+sE]`` is reconstructed as
   ``+a(0, sE) − a(0, sA)``.
-  ``from-previous-step`` — the archive stores per-step increments
   (``a(step − period, step)``). The window is a single interval.

The ``covering:`` key used for archive accumulations is **not** used
here: the covering is determined entirely by the basetime and the
``accumulation`` flag.

.. code:: yaml

   base_dates: {start: 2021-01-01, end: 2021-01-03, frequency: 12h}
   steps:      {start: 6, end: 30, frequency: 3h}

   input:
     join:
       - mars: {type: fc, class: od, param: [q, t], levtype: pl,
                level: [50], stream: oper, grid: 20./20., expver: "0001"}
       - pipe:
           - accumulate:
               period: 1h
               accumulation: from-zero
               source:
                 mars: {type: fc, class: od, param: [tp], levtype: sfc,
                        stream: oper, grid: 20./20., expver: "0001"}
           - rename:
               param: {tp: tp_accum_1h}

   output:
     layout: trajectories

Chunking
========

The default chunking for a trajectory output is
``{base_dates: 1, steps: 1, ensembles: 1}``. The ``variables`` and
``cells`` axes are stored full-length by default. Any key in the
``chunking`` dictionary that is not a dimension of the output raises an
error, so typos surface at build time.

On-disk layout
==============

The Zarr store contains, in addition to the usual coordinate and
metadata arrays:

-  ``data`` of shape ``(base_dates, variables, ensembles, steps,
   cells)`` and dimensions ``("time", "variable", "ensemble", "step",
   "cell")``;
-  ``base_dates`` — the forecast initialisation times;
-  ``steps`` — the forecast lead times (``numpy.timedelta64``);
-  ``latitudes`` and ``longitudes`` — unchanged from the gridded layout.

Attributes: ``layout = "trajectories"``, ``ensemble_dimension = 2``,
``step_dimension = -2``, plus ``start_date`` / ``end_date`` set to the
validity-time envelope (``min(basetime) + first_step`` to
``max(basetime) + last_step``).

.. _trajectories-using:

*******
 Using
*******

Open a trajectory dataset exactly like any other Anemoi dataset:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("path/to/trajectory.zarr")

Trajectory-specific attributes
==============================

.. code:: python

   ds.base_dates        # Forecast initialisation times
   ds.steps             # Forecast lead times (numpy.timedelta64)

   ds.base_start_date   # First base date
   ds.base_end_date     # Last base date
   ds.base_frequency    # Base-date interval

   ds.step_start        # First step (datetime.timedelta)
   ds.step_end          # Last step
   ds.step_frequency    # Step interval, or None if steps are not uniform

   ds.start_date        # Envelope: first base date + first step
   ds.end_date          # Envelope: last  base date + last  step

``ds.dates`` is an alias for ``ds.base_dates`` (to keep shared Anemoi
code working unchanged). ``ds.frequency`` is **not** available — use
``ds.base_frequency`` or ``ds.step_frequency`` depending on which axis
you need.

``ds.shape`` is 5-D: ``(base_dates, variables, ensembles, steps,
cells)``.

Selecting along the step axis
=============================

``open_dataset`` accepts additional keyword arguments to subset the step
axis:

-  ``step=<int>`` — select a single forecast step; returns a
   :ref:`gridded-like<layouts-gridded>` 4-D view ``(base_dates,
   variables, ensembles, cells)`` through a ``SingleStepView`` wrapper.
-  ``steps=[<int>, …]`` — select a list of steps; keeps the 5-D shape
   with the step axis narrowed (``StepSubset``).
-  ``step_start``, ``step_end``, ``step_frequency`` — range form of the
   above.
-  ``base_start``, ``base_end`` — filter the base-date axis without the
   valid-time envelope logic.

.. code:: python

   # One forecast step — looks like a gridded dataset
   ds_t6 = open_dataset("traj.zarr", step=6)

   # A range of steps
   ds_short = open_dataset("traj.zarr", step_start=6, step_end=18,
                           step_frequency="6h")

   # Filter by base date
   ds_202101 = open_dataset("traj.zarr",
                            base_start="2021-01-01",
                            base_end="2021-01-31")

The ``start=`` / ``end=`` kwargs also work and use *envelope*
semantics: a base date is kept if and only if its full step range
``[base + step_start, base + step_end]`` lies inside ``[start, end]``.

Inspecting
==========

``anemoi-datasets inspect`` detects the trajectory layout automatically
and prints the base-date and step axes separately.

.. code:: console

   $ anemoi-datasets inspect trajectory.zarr
