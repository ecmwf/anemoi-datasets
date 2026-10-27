.. _grid-observations-source:

###################
 grid_observations
###################

This source turns observations (rows with a time and a position) into
gridded fields: for each dataset date *T*, every grid cell holds one
observation of *T*'s time window, or the mean of one overpass, plus
derived geometry and forcings. Cells without an observation are
missing (NaN), so set ``build.allow_nans: true``.

It wraps a tabular ``input`` (a source such as :ref:`parquet
<parquet-source>` or :ref:`odb <odb-source>`, or a ``pipe`` of one and
tabular filters). The gridding is ``anemoi.transform.obs``, which does no
I/O, so real-time processing can share it.

.. literalinclude:: yaml/grid-observations.yaml
   :language: yaml

********
 Window
********

``window`` is relative to *T*, in the ``anemoi.utils`` window syntax:
``"(-350m,+10m]"`` holds the rows with ``T - 5h50 < time <= T + 10min``.
The dataset ``frequency`` is the step between windows. Offsets take a
single unit (``350m``, ``6h``) or ``H:M`` (``-5:50``); ``-5h50m`` is not
valid.

For each group of dates, the inner input is called once with a
``TimeSpan`` covering every window of the group, from the first window's
start to the last window's end. A window that reaches into the previous
day therefore needs no extra configuration.

***********
 Selection
***********

``select`` decides what fills a cell:

**candidates**
   ``latest_pass`` keeps the cell's observations within ``pass_gap``
   (default ``15min``) of its latest one, i.e. one overpass.
   ``window`` keeps every observation of the window.

**reduce**
   ``nearest`` takes the candidate nearest the cell centre (ties: the
   later). ``latest`` takes the latest (ties: the nearer). Both keep one
   whole row, so every channel comes from the same pixel. ``mean``
   averages the candidates of the latest observation's group, e.g.
   satellite, which is a superob. ``mean`` needs ``latest_pass``.

**mean_of**
   With ``reduce: mean``, ``radiance`` averages the channels as Planck
   radiance and stores the brightness temperature of the mean. It needs
   ``variables.wavenumbers``. The default ``bt`` averages the brightness
   temperatures.

**min_obs**
   Cells with fewer candidates stay empty (default 1).

Nearest cells are found on the unit sphere, so cells at the dateline and
the poles are exact.

****
 QC
****

``qc`` is applied in this order. Every input row is counted, so the
monitoring plots show what each step removes.

**flags**
   Keep the rows whose column is one of the values, e.g.
   ``{quality_flag: [0]}``.

**range**
   Values outside ``[lo, hi]`` become missing.

**known_bad**
   A list of ``{columns, groups, start, end, reason}``. Those values
   become missing.

**require_complete**
   Rows with any channel missing are not selected (default ``true``).

Any tabular filter can also be applied in the inner ``pipe``, before
this QC.

***********
 Variables
***********

**channels**
   Observed values, checked by ``require_complete``.

**columns**
   Other input columns, gridded as they are.

**derived**
   Computed from the selected observation:

   -  ``time_delta`` and ``time_delta_norm``: the observation time minus
      *T*, in hours and in dataset steps.
   -  ``cos_vza``, ``scan_pos`` and ``signed_sin_vza``: viewing
      geometry. The last two need ``scan``.
   -  ``offset_east_km``, ``offset_north_km`` and their ``_norm``
      versions: the position relative to the cell centre.
   -  ``n_obs``: the number of candidates.
   -  ``cos_sza_obs``: the observation's own solar zenith.

**forcings**
   Known functions of (*T*, grid point), filled in every cell: e.g.
   ``cos_latitude``, ``sin_julian_day``, ``cos_local_time``,
   ``cos_sza``.

**spread**
   Channels whose standard deviation over the candidates is stored as
   ``<name>_std``.

**rename**
   Output names. A key ending in ``_`` renames a prefix
   (``bt_: obsvalue_rawbt_``).

**group_by** (default ``satid``)
   The integer column that QC counts, statistics and ``mean`` superobs
   are kept per.

**maps**
   The output variables that ``anemoi-datasets monitor`` maps (see
   :ref:`monitor_command`).

*************
 Diagnostics
*************

While building, the source records, per group:

-  ``obs_stats``: count, sum, sum of squares, min and max of the
   selected values per day, group and variable;
-  ``obs_qc``: the rows each QC step removes, per group;
-  ``obs_qc_cells``: rows read and rejected per grid cell.

``finalise`` merges them into ``<dataset>/_monitor/data/``, where
``anemoi-datasets monitor`` plots them.

The grid's ``resolution`` (e.g. ``O96``) is stored with the dataset. The
recipe, including the window, is stored in the dataset's ``recipe``
attribute.
