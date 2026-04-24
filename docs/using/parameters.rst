.. _open_dataset-parameters:

#########################
 Open dataset parameters
#########################

This page is intended to provide level of support/applicability of the
various parameters that can be passed to the `open_dataset` function
when dealing with the different :ref:`dataset layouts
<layouts-introduction>`. Tabular and trajectories datasets are recent
additions to the *anemoi-datasets* package and are still under
development. The meaning of the emojis, applied per layout column, is
as follows:

-  ✅: Parameter has been tested and works for this layout.

-  🔁: Parameter will work, but has a slightly different meaning or
   behaviour for this layout.

-  ❌: Parameter is not applicable to this layout.

-  ❓: Parameter may work, but has not been tested with this layout.

-  🧪: Parameter should work, but has not been tested with this layout.

-  ⚠️: Parameter may work, but the behaviour is not fully understood.

-  🛠️: Will work in the future. Not yet implemented or tested, but
   expected to work without major issues.

-  🗑️: Is obsolete and will be removed in the future. Should not be used
   for new code, but may still work for now. Will be removed in a future
   release.

.. warning::

   Datasets of different layouts cannot be combined. So `concat`,
   `join`, and similar operations are not supported between layouts.
   The table below assumes that when combining datasets, they have the
   same layout. See :ref:`layouts-introduction` for more information.

.. list-table:: Parameters for ``open_dataset``

   -  -  parameter
      -  gridded
      -  tabular
      -  trajectories
      -  comment

   -  -  adjust

      -  ✅

      -  ⚠️

      -  ⚠️

      -  Adjustment mode when combining datasets, e.g. select common
         dates, variables, etc. This needs testing and possibly decision
         on expected behaviour for tabular and trajectory datasets.

   -  -  area
      -  ✅
      -  ❌
      -  🧪
      -  Spatial cropping area as a list [lon_min, lat_min, lon_max,
         lat_max].

   -  -  base_end
      -  ❌
      -  ❌
      -  ✅
      -  Trajectories only: filter the base-date axis directly (no
         valid-time envelope logic). See :ref:`subsetting-trajectories`.

   -  -  base_start
      -  ❌
      -  ❌
      -  ✅
      -  Trajectories only: filter the base-date axis directly (no
         valid-time envelope logic). See :ref:`subsetting-trajectories`.

   -  -  chain

      -  🗑️

      -  🗑️

      -  🗑️

      -  Experimental chain operation. Same behaviour as `concat`, but
         does not check that the dates are continous. Will be removed in
         the future.

   -  -  concat

      -  ✅

      -  ⚠️

      -  🧪

      -  Concatanate two or more datasets along the time dimension. That
         may work, but the behaviour of the windowing at the seam is not
         well defined. So it should be skipped for now.

   -  -  complement
      -  ✅
      -  ❌
      -  🧪
      -  Complement/cutout configuration (used for creating
         complements).

   -  -  cutout
      -  ✅
      -  ❌
      -  🧪
      -  List of datasets used as cutouts for complements/cutout
         operations.

   -  -  drop
      -  ✅
      -  🧪
      -  🧪
      -  Variables to drop (list).

   -  -  end

      -  ✅

      -  🔁

      -  🔁

      -  Set the end date for the opened dataset. For gridded datasets,
         the date must be present in the dataset. For tabular datasets,
         the date is used as-is, and any windows requested between the
         actual end of date of the data and that date will return empty
         arrays (See :ref:`layouts-tabular`). For trajectories, the
         date applies to the validity-time envelope: a base date is
         kept iff ``[base + step_start, base + step_end]`` lies inside
         ``[start, end]`` (See :ref:`layouts-trajectories`). Use
         ``base_end`` to filter the base-date axis directly.

   -  -  ensemble
      -  ✅
      -  ❌
      -  🧪
      -  List of datasets forming an ensemble (e.g. ``ensemble=[d1,
         d2]``).

   -  -  fill_missing_dates
      -  ✅
      -  ❌
      -  ❌
      -  Method to fill missing dates ("interpolate" or "closest").
         Trajectories have no missing-date concept.

   -  -  fill_missing_gaps
      -  ✅
      -  ❌
      -  ❌
      -  Fill virtual datasets for gaps when concatenating.

   -  -  frequency

      -  ✅

      -  🔁

      -  🔁

      -  For gridded dataset, select the frequency of the return sample;
         it must be a multiple of the dataset frequency. For tabular
         datasets, it is used to create windows of the specified
         frequency (e.g. "1D" for daily windows) and is not connected to
         the dataset frequency (which is undefined) (See
         :ref:`layouts-tabular`). For trajectories, ``frequency`` acts
         on the base-date axis; use ``step_frequency`` for the step
         axis (See :ref:`layouts-trajectories`).

   -  -  grids
      -  ✅
      -  ❌
      -  🧪
      -  List of grids/datasets to combine as multiple grid sources.

   -  -  interpolate_frequency
      -  ✅
      -  ❌
      -  🧪
      -  Frequency used to interpolate a dataset to a higher temporal
         resolution.

   -  -  interpolate_variables
      -  ✅
      -  ❌
      -  🧪
      -  Variables to interpolate spatially (with optional
         ``max_distance``).

   -  -  interpolation
      -  ✅
      -  ❌
      -  🧪
      -  Interpolation method (example: "nearest").

   -  -  join
      -  ✅
      -  🧪
      -  🧪
      -  Join two or more datasets along the variable dimension.

   -  -  max_distance
      -  ✅
      -  ❌
      -  🧪
      -  Maximum distance used by spatial interpolation (e.g.
         nearest-neighbour).

   -  -  member, members
      -  ✅
      -  ❌
      -  🧪
      -  0-based member selection (see `number` for 1-based selection).

   -  -  merge
      -  ✅
      -  ❌
      -  🧪
      -  Merge operation key to combine datasets by overlaying fields.

   -  -  name

      -  ✅

      -  ✅

      -  ✅

      -  Experimental. Optional name assigned to the resulting dataset
         subset that can be used to name masks that will be retrieved in
         inference.

   -  -  number, numbers
      -  ✅
      -  ❌
      -  🧪
      -  1-based member selection (see `member` for 0-based selection).

   -  -  reorder
      -  ✅
      -  🧪
      -  🧪
      -  Reorder variables (list or mapping).

   -  -  rename
      -  ✅
      -  🧪
      -  🧪
      -  Rename variables mapping.

   -  -  rescale
      -  ✅
      -  🛠️
      -  🧪
      -  Rescaling mapping/tuples/units for variables.

   -  -  select
      -  ✅
      -  ✅
      -  🧪
      -  Select variables (list, set or string).

   -  -  set_missing_dates
      -  ✅
      -  ❌
      -  ❌
      -  Debug option: list of dates to mark as missing.

   -  -  shuffle
      -  🗑️
      -  🗑️
      -  🗑️
      -  Boolean to shuffle dataset indices when subsetting.

   -  -  skip_missing_dates
      -  ✅
      -  ❌
      -  ❌
      -  Boolean: skip missing dates when iterating (requires
         ``expected_access``). Trajectories have no missing-date
         concept.

   -  -  source
      -  ✅
      -  ❌
      -  🧪
      -  Source dataset name/path used in complement examples.

   -  -  start

      -  ✅

      -  🔁

      -  🔁

      -  Set the start date for the opened dataset. For gridded
         datasets, the date must be present in the dataset. For tabular
         datasets, the date is used as-is, and any windows requested
         between that date and the actual start of date of the data will
         return empty arrays (See :ref:`layouts-tabular`). For
         trajectories, ``start`` applies to the validity-time envelope
         (see ``end``). Use ``base_start`` to filter the base-date axis
         directly.

   -  -  statistics
      -  ✅
      -  🧪
      -  🧪
      -  Use the statistics of another dataset.

   -  -  step
      -  ❌
      -  ❌
      -  ✅
      -  Trajectories only: select a single forecast step. Returns a
         4-D view ``(base_dates, variables, ensembles, cells)`` —
         shape-compatible with a gridded dataset at that lead time. See
         :ref:`subsetting-trajectories`.

   -  -  step_end
      -  ❌
      -  ❌
      -  ✅
      -  Trajectories only: end of a step-axis range selection.

   -  -  step_frequency
      -  ❌
      -  ❌
      -  ✅
      -  Trajectories only: frequency for a step-axis range selection.

   -  -  step_start
      -  ❌
      -  ❌
      -  ✅
      -  Trajectories only: start of a step-axis range selection.

   -  -  steps
      -  ❌
      -  ❌
      -  ✅
      -  Trajectories only: list of forecast steps to retain. Keeps the
         5-D shape with the step axis narrowed. See
         :ref:`subsetting-trajectories`.

   -  -  thinning
      -  ✅
      -  ✅
      -  🧪
      -  Thinning factor or proportion.

   -  -  trim_edge
      -  ✅
      -  ❌
      -  🧪
      -  Tuple to trim edges of the grid (e.g. ``(1,2,3,4)``).

   -  -  window

      -  ❌

      -  ✅

      -  ❌

      -  Window specification for tabular datasets. For gridded and
         trajectory datasets, that parameter is ignored. See
         :ref:`layouts-tabular` for details.

   -  -  x
      -  🗑️
      -  🗑️
      -  🗑️
      -  Experimental: x coordinate for `xy` selection.

   -  -  xy
      -  🗑️
      -  🗑️
      -  🗑️
      -  Experimental xy selection mode.

   -  -  y
      -  🗑️
      -  🗑️
      -  🗑️
      -  Experimental: y coordinate for `xy` selection.

   -  -  zip
      -  🗑️
      -  🗑️
      -  🗑️
      -  Experimental zip mode to combine datasets.
