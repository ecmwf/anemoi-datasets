.. _open_dataset-parameters:

#########################
 Open dataset parameters
#########################

This page lists keyword parameters to ``open_dataset()`` found in the
documentation (examples in ``.rst`` and ``.py`` files).

.. list-table:: Parameters for ``open_dataset`` :header-rows: 1 :widths: 20 10 10 60

   -  -  parameter
      -  gridded
      -  tabular
      -  comment

   -  -  adjust
      -  ✅
      -  ❌
      -  Adjustment mode when combining datasets (example: "all").

   -  -  area
      -  ✅
      -  ❌
      -  Spatial cropping area as a list [lon_min, lat_min, lon_max,
         lat_max].

   -  -  chain
      -  ✅
      -  ❌
      -  Experimental chain operation (internal use).

   -  -  concat
      -  ✅
      -  ❌
      -  List of datasets to concatenate (e.g. ``concat=[d1, d2]``).

   -  -  complement
      -  ✅
      -  ❌
      -  Complement/cutout configuration (used for creating
         complements).

   -  -  cutout
      -  ✅
      -  ❌
      -  List of datasets used as cutouts for complements/cutout
         operations.

   -  -  dataset
      -  ✅
      -  ❌
      -  Dataset identifier or a dict describing join/merge/config.

   -  -  datasets
      -  ✅
      -  ❌
      -  Alternative plural key for passing multiple dataset entries.

   -  -  drop
      -  ✅
      -  ❌
      -  Variables to drop (list).

   -  -  end
      -  ✅
      -  ❌
      -  End date for the opened dataset (string or date-like).

   -  -  ensemble
      -  ✅
      -  ❌
      -  List of datasets forming an ensemble (e.g. ``ensemble=[d1,
         d2]``).

   -  -  fill_missing_dates
      -  ✅
      -  ❌
      -  Method to fill missing dates ("interpolate" or "closest").

   -  -  fill_missing_gaps
      -  ✅
      -  ❌
      -  Fill virtual datasets for gaps when concatenating.

   -  -  frequency
      -  ✅
      -  ❌
      -  Frequency of the dataset (e.g. "3h", "24h") or used in join
         entries.

   -  -  grids
      -  ✅
      -  ❌
      -  List of grids/datasets to combine as multiple grid sources.

   -  -  interpolate_frequency
      -  ✅
      -  ❌
      -  Frequency used to interpolate a dataset to a higher temporal
         resolution.

   -  -  interpolate_variables
      -  ✅
      -  ❌
      -  Variables to interpolate spatially (with optional
         ``max_distance``).

   -  -  interpolation
      -  ✅
      -  ❌
      -  Interpolation method (example: "nearest").

   -  -  join
      -  ✅
      -  ❌
      -  Nested list/dict describing datasets to join (used inside
         ``dataset={"join": [...]}``).

   -  -  k
      -  ✅
      -  ❌
      -  Integer parameter used alongside interpolation/complement
         examples (e.g. ``k=1``).

   -  -  max_distance
      -  ✅
      -  ❌
      -  Maximum distance used by spatial interpolation (e.g.
         nearest-neighbour).

   -  -  member
      -  ✅
      -  ❌
      -  0-based member selection (alias of ``members``).

   -  -  members
      -  ✅
      -  ❌
      -  Alias for ``member`` (plural form).

   -  -  method
      -  ✅
      -  ❌
      -  Method used for thinning or other operations (e.g. "grid",
         "random").

   -  -  merge
      -  ✅
      -  ❌
      -  Merge operation key to combine datasets by overlaying fields.

   -  -  min_distance_km
      -  ✅
      -  ❌
      -  Minimum distance in kilometres used when computing
         complements/cutouts.

   -  -  name
      -  ✅
      -  ❌
      -  Optional name assigned to the resulting dataset subset.

   -  -  number
      -  ✅
      -  ❌
      -  1-based member selection.

   -  -  numbers
      -  ✅
      -  ❌
      -  Alias for ``number`` (plural form).

   -  -  padding
      -  ✅
      -  ❌
      -  Padding configuration used when creating a `Padded` subset.

   -  -  reorder
      -  ✅
      -  ❌
      -  Reorder variables (list or mapping).

   -  -  rename
      -  ✅
      -  ❌
      -  Rename variables mapping.

   -  -  rescale
      -  ✅
      -  ❌
      -  Rescaling mapping/tuples/units for variables.

   -  -  select
      -  ✅
      -  ❌
      -  Select variables (list, set or string).

   -  -  set_missing_dates
      -  ✅
      -  ❌
      -  Debug option: list of dates to mark as missing.

   -  -  shuffle
      -  ✅
      -  ❌
      -  Boolean to shuffle dataset indices when subsetting.

   -  -  skip_missing_dates
      -  ✅
      -  ❌
      -  Boolean: skip missing dates when iterating (requires
         ``expected_access``).

   -  -  source
      -  ✅
      -  ❌
      -  Source dataset name/path used in complement examples.

   -  -  start
      -  ✅
      -  ❌
      -  Start date for the opened dataset (string or date-like).

   -  -  thinning
      -  ✅
      -  ❌
      -  Thinning factor or proportion (integer for km or float for
         fraction).

   -  -  trim_edge
      -  ✅
      -  ❌
      -  Tuple to trim edges of the grid (e.g. ``(1,2,3,4)``).

   -  -  tensors
      -  ✅
      -  ❌
      -  Specification for returning tensors instead of arrays.

   -  -  x
      -  ✅
      -  ❌
      -  Experimental: x coordinate for `xy` selection.

   -  -  xy
      -  ✅
      -  ❌
      -  Experimental xy selection mode.

   -  -  y
      -  ✅
      -  ❌
      -  Experimental: y coordinate for `xy` selection.

   -  -  zip
      -  ✅
      -  ❌
      -  Experimental zip mode to combine datasets.
