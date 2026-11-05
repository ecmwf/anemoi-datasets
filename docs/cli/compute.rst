.. _compute_command:

Compute Command
===============

The ``compute`` command recomputes statistics, statistics of temporal
*tendencies*, or statistics of the *residual* between two datasets, **on the
fly** from an opened dataset, and can also **generate a new dataset** from the
values it reads. It is deliberately standalone (it does not reuse the
creation-time statistics code) and runs as a single, simple chunked loop with
optional parallelism. All accumulation is done in ``float64`` using a
numerically-stable (parallel/Welford) algorithm, so precision is preserved even
over very large datasets.

Typical uses:

- Re-derive the statistics of a dataset *as opened* (with ``select``, ``start``,
  ``end``, sub-area, rescaling, ... applied) without rewriting the Zarr store.
- Compute tendency statistics for an arbitrary time delta.
- Compare two datasets at different resolutions by interpolating both onto a
  common grid (``--grid``) and computing the statistics of their difference
  (``--minus``).
- Validate a dataset's stored statistics with ``--compare``.
- Materialise a view -- a subset, a selection, a join, an interpolation to
  another grid, ... -- or the difference between two datasets as a dataset of its
  own (``--output``), with the recomputed statistics as its statistics.

Synopsis
--------

.. code-block:: bash

    anemoi-datasets compute <dataset> \
        [--statistics] [--statistics-tendencies 6h] \
        [--minus <dataset-2>] [--grid o96] [--grid-method nearest] \
        [--start DATE] [--end DATE] [--frequency FREQ] [--output PATH.zarr] \
        [--chunk-size N] [--sample-dates FRACTION] [--compare] \
        [--output-statistics FILE.npz] [--overwrite] [--checkpoint PATH] [--resume] [--parallel N]

While the command runs it shows a progress bar and, in an interactive terminal,
refreshes a statistics table (the same columns as ``inspect``) for all variables
every ten seconds, so the running values can be eyeballed. Each numeric cell also
shows the signed relative change since the previous refresh, e.g. ``0.707 (+0.14%)``;
when the baseline is zero or the change exceeds +/-100% it falls back to the
absolute delta, e.g. ``0.707 (+0.001)``.

The dataset
~~~~~~~~~~~

``<dataset>`` (and the dataset after ``--minus``) can be given in **two ways**:

- **Name/path with options**: a dataset name or path optionally followed by
  ``key=value`` tokens that are forwarded to ``open_dataset`` (e.g.
  ``my-dataset start=2020-01-01 end=2020-12-31 select=2t``). Values are coerced to
  ``int``/``float``/``bool``/``None`` when possible, otherwise kept as strings (so
  dates such as ``2020-01-01`` stay strings).

- **A single JSON literal**: a complete ``open_dataset`` configuration, passed
  straight to ``open_dataset`` (e.g.
  ``'{"dataset": "x", "start": "2020-01-01", "select": ["2t"]}'``). This is
  convenient for complex, nested configurations.

.. important::

    The JSON is **only** an ``open_dataset`` config. The compute options
    (``--statistics``, ``--statistics-tendencies``, ``--parallel``, ``--output-statistics``,
    ...) are **always** CLI flags and must never be put inside the JSON. Mixing
    ``key=value`` options with a JSON dataset is rejected — put every
    ``open_dataset`` option inside the JSON in that case.

If neither ``--statistics`` nor ``--statistics-tendencies`` is given, plain statistics are
computed by default. NaNs are ignored on a per-variable basis by default.

Missing dates
~~~~~~~~~~~~~

The dates a dataset records as missing cannot be read, so they are left out of
the computation and a warning reports how many were skipped. With ``--minus``, a
date missing from *either* dataset is skipped, since the two are subtracted by
position. Open the dataset with ``fill_missing_dates`` to fill and include them
instead.

Tendencies are never computed across a gap: ``--statistics-tendencies`` uses only
the pairs of dates that are ``DELTA`` apart with no missing date in between. The
result is the same sequentially and with ``--parallel``.

``--output`` carries the missing dates over: they stay ``NaN`` in the
generated store and are recorded in its ``missing_dates``, so it is a dataset
with the same gaps.

Options
-------

``--statistics``
    Compute the plain statistics (mean, minimum, maximum, stdev).

``--statistics-tendencies DELTA``
    Compute statistics of the tendencies ``value(t) - value(t - delta)`` for a
    single delta (e.g. ``6h``). The delta must be a whole multiple of the dataset
    frequency.

``--minus <dataset-2> [key=value ...]``
    Subtract ``dataset-2`` from the dataset: everything downstream (the
    statistics, the tendencies, the generated dataset) then describes
    ``dataset - dataset-2``. The subtraction is value by value, by position, so
    the two datasets must have the same dates, the same variables in the same
    order and the same field shape once their respective ``open_dataset`` options
    are applied. Their grids may differ only if ``--grid`` is given, which
    interpolates both onto a common grid first. ``<dataset-2>`` can be a name
    with ``key=value`` options or a single JSON ``open_dataset`` config.

    .. warning::

        Without ``--grid``, only the *shapes* are checked, not the coordinates.
        Two different grids with the same number of points are subtracted point by
        point, which is meaningless. Pass ``--grid`` whenever the two datasets are
        not known to be on exactly the same grid.

``--grid GRID``
    Interpolate the dataset -- and the one given to ``--minus`` -- onto this grid
    before anything else (see `Interpolating to a grid`_). ``GRID`` is a named
    grid (``o96``, ``n320``, ...), a resolution (``0.25``, ``0.25x0.25``) or the
    path of an ``.npz`` file holding ``latitudes`` and ``longitudes`` arrays.

``--grid-method METHOD``
    The interpolation method used by ``--grid`` (default: ``nearest``). Requires
    ``--grid``.

``--start DATE``, ``--end DATE``, ``--frequency FREQ``
    ``open_dataset`` options applied to **every** dataset of the command, so that
    the dataset and the one given to ``--minus`` do not have to repeat them --
    which they nearly always must, since the two are subtracted by position and
    therefore have to be on the same dates::

        anemoi-datasets compute a --minus b --start 2023-05-12 --end 2023-05-29

    is the same as::

        anemoi-datasets compute a start=2023-05-12 end=2023-05-29 \
            --minus b start=2023-05-12 end=2023-05-29

    They also reach a dataset given as a JSON config, which is the one case where
    an ``open_dataset`` option can be a flag. Giving the same option both ways --
    as the flag and as a ``key=value`` or inside the JSON -- is an error rather
    than a silent winner.

``--chunk-size N``
    Number of time steps read per chunk (default: 1).

``--sample-dates FRACTION``
    Compute over only a random fraction of the dates (e.g. ``0.1`` for 10%). The
    sample is deterministic (seeded from the arguments) so a resumed run is
    consistent. Not compatible with ``--statistics-tendencies`` (tendencies need
    adjacent dates) nor with ``--parallel``.

``--compare``
    Compare the recomputed statistics (and tendencies) against the dataset's
    stored ``statistics`` / ``statistics_tendencies(delta)`` and print the
    absolute and relative differences. Not applicable to ``--minus``.

``--output-statistics FILE.npz``
    Write the statistics (and any ``--compare`` differences) to this compressed
    numpy archive (see `The statistics file`_). The path must end with ``.npz``,
    which is checked before anything is computed. They are always written:
    without ``--output-statistics`` the default path is
    ``<dataset-name>.statistics.npz`` in the current directory.

``--output PATH.zarr``
    Also write the values read by the loop to a new Zarr dataset (see
    `Generating a dataset`_). The path must end with ``.zarr``. Its ``data``
    array is ``float32``, as in ``anemoi-datasets create``; the statistics are
    always accumulated and stored in ``float64``.

``--overwrite``
    Replace the statistics file (and the generated dataset) if either already
    exists. Without it the command fails immediately, before computing anything.

``--parallel N``
    Compute using ``N`` worker processes. The time range is split into segments
    computed independently and merged. Tendency segments are seeded with the
    ``delta`` rows before their start, so boundary tendencies remain exact; the
    parallel result is identical to the sequential one.

``--checkpoint PATH``
    Path of the checkpoint file. Defaults to
    ``./compute-checkpoint-<sha1>.pkl``, where ``<sha1>`` is a hash of the
    arguments that affect the result. A checkpoint is written roughly every
    minute (sequential) or after every completed segment (parallel), and is
    removed automatically on successful completion.

``--resume``
    Resume an interrupted computation from its checkpoint. The arguments must
    match those used when the checkpoint was created (verified via the hash);
    otherwise the command errors out.

Examples
--------

Recompute the statistics of a dataset as opened over a sub-period:

.. code-block:: bash

    anemoi-datasets compute my-dataset start=2020-01-01 end=2020-12-31 --statistics

Compute 6-hour tendency statistics in parallel and save to JSON:

.. code-block:: bash

    anemoi-datasets compute my-dataset --statistics-tendencies 6h --parallel 8 --output-statistics tend.npz

Estimate statistics quickly from 10% of the dates:

.. code-block:: bash

    anemoi-datasets compute my-dataset --statistics --sample-dates 0.1

Validate a dataset's stored statistics:

.. code-block:: bash

    anemoi-datasets compute my-dataset --statistics --compare

Give the dataset as a JSON ``open_dataset`` config (compute options stay as flags):

.. code-block:: bash

    anemoi-datasets compute '{"dataset": "my-dataset", "start": "2020-01-01",
        "end": "2020-12-31", "select": ["2t", "10u", "10v"]}' --statistics --parallel 8

Statistics of the difference between a high-resolution dataset and a
low-resolution one, both interpolated onto the same grid:

.. code-block:: bash

    anemoi-datasets compute hi-res --minus lo-res --grid o96

Resume a long run that was interrupted:

.. code-block:: bash

    anemoi-datasets compute my-dataset --statistics --parallel 16 --resume

Interpolating to a grid
-----------------------

``--grid`` brings the dataset -- and, with ``--minus``, the dataset subtracted
from it -- onto a common grid before anything else happens. Everything downstream
(the statistics, the tendencies, the generated dataset) then describes the values
on that grid:

.. code-block:: bash

    # statistics of the difference between two resolutions
    anemoi-datasets compute hi-res --minus lo-res --grid o96 --statistics

    # a dataset regridded to o96
    anemoi-datasets compute hi-res --grid o96 --output hi-res-o96.zarr

The interpolation itself is done by anemoi-transform, the same code its
``regrid`` filter uses:

``nearest`` (the default)
    Uses :func:`anemoi.transform.spatial.nearest_grid_points`: a KD-tree over the
    source points, then each target point takes the value of its nearest source
    point. It needs nothing but the latitudes and longitudes of the two grids, so
    it works for any pair of grids, named or not.

Any other method
    Passed to ``earthkit.regrid.interpolate``, field by field. This needs both
    grids to be known to earthkit-regrid and a pre-generated matrix to exist for
    the ``(in_grid, out_grid, method)`` triple; the input grid is taken from the
    dataset's ``resolution`` attribute, and the command fails with a clear message
    when the dataset has none.

The target grid is resolved by :func:`anemoi.transform.grids.named.lookup`. Named
grids are downloaded once from the anemoi grid registry and cached locally; an
``.npz`` path is read directly and must hold ``latitudes`` and ``longitudes``
arrays.

A dataset that is already on the target grid is left alone -- no interpolation
pass, no cost -- so ``--grid`` on a pair where one dataset is already on the
target grid only interpolates the other one.

A generated dataset then carries the target grid: its ``latitudes``,
``longitudes`` and ``field_shape`` are those of the grid, its ``resolution`` is
the grid name, and ``data_request`` and ``proj_string``, which describe the
source grid, are not propagated.

Generating a dataset
--------------------

``--output PATH.zarr`` writes the values read by the loop -- the dataset
*as opened*, or the residual ``<dataset> - <dataset-2>`` -- into a new Zarr
dataset, in the same single pass as the statistics:

.. code-block:: bash

    anemoi-datasets compute my-dataset select=2t start=2020-01-01 end=2020-12-31 \
        --output 2t-2020.zarr

    anemoi-datasets compute forecast --minus analysis --grid o96 \
        --output forecast-error.zarr --parallel 16

The store is written by the same code as ``anemoi-datasets create``, so it holds
the usual ``data``, ``latitudes``, ``longitudes`` and statistics arrays, with one
date and one ensemble member per chunk, and can be opened with ``open_dataset``
like any other dataset.

The generated dataset has the **layout of the one it came from**, which is
recognised from the rank of the view:

``gridded``
    ``(time, variable, ensemble, cell)``, with a ``dates`` array and a single
    ``frequency``.

``trajectories``
    ``(base date, variable, ensemble, step, cell)``, with ``base_dates`` and
    ``steps`` arrays, one base date *and one step* per chunk, ``frequency`` the
    base-date one, ``start_base_date`` / ``end_base_date`` for the base-date
    axis, and ``start_date`` / ``end_date`` the valid-time envelope
    ``base date + step``.

Nothing else differs: the values are read, interpolated and differenced along the
same axes in both cases -- time is axis 0 and variables axis 1 either way -- so
the statistics, the constant-field detection and ``--parallel`` are common code.

Metadata
~~~~~~~~

There is no recipe behind a generated dataset, so its metadata is derived from the
opened dataset(s) instead:

- ``variables``, ``variables_metadata``, ``frequency``, ``start_date``,
  ``end_date``, ``resolution``, ``field_shape`` and the ``dates``, ``latitudes``
  and ``longitudes`` arrays come from the dataset *as opened* -- that is, after
  ``select``, ``start``/``end``, and so on -- or from the ``--grid`` grid for
  everything spatial when one is requested.
- ``mean``, ``stdev``, ``minimum`` and ``maximum`` are the statistics computed by
  this command, and ``statistics_tendencies_<delta>_*`` are written as well when
  ``--statistics-tendencies`` is given. Statistics are always computed for a
  generated dataset, even when only tendencies were requested.
- ``constant_fields`` lists the variables that are constant in time, detected
  while reading.
- ``licence``, ``attribution``, ``data_request``, ``origins``, ``proj_string``
  and ``variable_naming`` are copied from the source dataset(s) when they are
  present and unanimous. For a residual only ``proj_string`` and
  ``variable_naming`` are copied: the grid and the variable names survive a
  difference, the lineage of the values does not.
- ``derived_from`` records how the dataset was made: the ``open_dataset``
  arguments of each source, the metadata of each source dataset, the arithmetic
  applied (``datasets[0]`` or ``datasets[0] - datasets[1]``), the ``--grid`` and
  ``--grid-method`` used, and the command line.
- ``uuid`` is new, and ``version`` is the current dataset format version.

.. warning::

   For a residual, ``variables_metadata`` is copied from the first dataset. It
   describes the values of *that* dataset, not the differences that are stored:
   a property such as ``is_accumulation`` no longer applies.

Restrictions
~~~~~~~~~~~~

- The opened dataset must be **gridded** (``time``, ``variable``, ``ensemble``,
  ``cell``) or **trajectories** (``base date``, ``variable``, ``ensemble``,
  ``step``, ``cell``); the generated dataset has the same layout. With ``--grid``
  it is on the requested grid.
- Missing dates are carried over: they are not read, so they keep the ``NaN``
  the ``data`` array is created with, and they are recorded in the generated
  store's ``missing_dates``. The result is a dataset with the same gaps as the
  one(s) it came from, which raises ``MissingDateError`` on those dates like any
  other. Open the source with ``fill_missing_dates`` to fill them instead.
- ``--sample-dates`` cannot be used: every date has to be read to be written.
- ``--parallel`` is supported: segments are chunk-aligned and the output has one
  date per chunk, so no two workers write to the same chunk.
- ``--resume`` continues writing into the existing store, using the checkpoint to
  know which blocks are already there. A run interrupted *without* ``--resume``
  leaves a partially-written store, whose unwritten dates are ``NaN``.
- The name of the store is checked against the :ref:`naming conventions
  <naming-conventions>` and a warning is printed when it does not follow them.

The statistics file
-------------------

The statistics are written to the path given by ``--output-statistics``, which
**must end with** ``.npz``: a compressed numpy archive. Each statistic is one
``float64`` array, named ``statistics.mean``, ``statistics.stdev``, ... (and
``tendency_statistics.``... for the tendencies), and NaN stays NaN. Everything
else in the document -- the header, the variable names, the tendency delta, the
provenance, the comparison -- is kept as a JSON string in a single ``metadata``
entry, so the archive describes itself:

.. code-block:: python

    import json

    import numpy as np

    with np.load("residual.npz") as z:
        metadata = json.loads(str(z["metadata"]))
        mean = z["statistics.mean"]

    print(metadata["variables"], mean)

The document always starts with a header that identifies it:

``kind``
    ``"residual-statistics"`` when ``--minus`` was used,
    ``"statistics"`` otherwise.

``version``
    The format version.

``datasets``
    The dataset labels the statistics were computed from. For a residual, the
    two datasets in the order they were subtracted, so that the residual is
    ``datasets[0] - datasets[1]``; otherwise the single dataset.

``derived_from``
    How the file was produced: the ``open_dataset`` call(s) the values came from
    and the metadata of the datasets they returned, the ``arithmetic``
    (``datasets[0]`` or ``datasets[0] - datasets[1]``), the ``computation``
    settings (tendency, chunk size, NaN policy, grid and method, and the command
    line), the ``anemoi_datasets_version`` and the time it was ``created``. It is
    the same entry a dataset generated by ``--output`` carries, so the statistics
    file and the dataset written beside it describe themselves identically.

    .. note::

        ``derived_from`` has a ``version`` of its own, of the *provenance* entry.
        It is unrelated to the document's top-level ``version``, which is the
        format version.

    Files written before this entry existed do not have it, and are still read.

Then come ``variables`` and the ``statistics`` / ``tendency_statistics`` blocks,
each mapping ``mean``, ``stdev``, ``minimum`` and ``maximum`` to one array of one
value per variable, in the order given by ``variables``. The ``output_dataset``
entry holds
the path of the generated dataset, or ``null``, and ``grid`` the grid the values
were interpolated to, or ``null``.

A residual document can be fed straight back into ``open_dataset``:

.. code-block:: python

    ds = open_dataset("lo-res", residual_statistics="residual.npz")
    print(ds.residual_statistics)

.. warning::

   Experimental: the ``residual_statistics`` option and the
   ``residual_statistics`` attribute may be removed or renamed in a
   future release.

The ``kind`` marker is checked when the file is read, so a plain statistics file
passed as ``residual_statistics`` is rejected instead of being used by mistake.
See :ref:`using-statistics`.

Notes
-----

- The computation reads the dataset *as opened*, so any ``open_dataset`` option
  (selection, sub-area, rescaling, ...) is reflected in the recomputed statistics.
- ``--compare`` reads the dataset's stored statistics for the *full* opened
  dataset; if you restrict the period with ``start``/``end`` (or ``--sample-dates``)
  the recomputed values will legitimately differ from the stored ones.
- Trajectory datasets expose two frequencies -- one for the base dates and one
  for the forecast steps -- so there is no single "previous date" and
  ``--statistics-tendencies`` is refused for them. Everything else works: the
  values are read, interpolated and differenced along the same axes (variables
  are axis 1 in both layouts), and ``--output`` generates a trajectories
  store, with its ``base_dates`` and ``steps`` arrays, its two frequencies and
  the valid-time envelope ``base date + step`` as its ``start_date`` /
  ``end_date``.
