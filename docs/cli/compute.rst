.. _compute_command:

Compute Command
===============

The ``compute`` command recomputes statistics, statistics of temporal
*tendencies*, or statistics of the *residual* between two datasets, **on the
fly** from an opened dataset. It is deliberately standalone (it does not reuse the
creation-time statistics code) and runs as a single, simple chunked loop with
optional parallelism. All accumulation is done in ``float64`` using a
numerically-stable (parallel/Welford) algorithm, so precision is preserved even
over very large datasets.

Typical uses:

- Re-derive the statistics of a dataset *as opened* (with ``select``, ``start``,
  ``end``, sub-area, rescaling, ... applied) without rewriting the Zarr store.
- Compute tendency statistics for an arbitrary time delta.
- Compare a regridded dataset against a reference at a different resolution by
  computing the statistics of their difference (``--statistics-residual``).
- Validate a dataset's stored statistics with ``--compare``.

Synopsis
--------

.. code-block:: bash

    anemoi-datasets compute <dataset> \
        [--statistics] [--statistics-tendencies 6h] \
        [--statistics-residual <dataset-2>] \
        [--chunk-size N] [--sample-dates FRACTION] [--compare] \
        [--output FILE.json] [--overwrite] [--checkpoint PATH] [--resume] [--parallel N]

While the command runs it shows a progress bar and, in an interactive terminal,
refreshes a statistics table (the same columns as ``inspect``) for all variables
every ten seconds, so the running values can be eyeballed. Each numeric cell also
shows the signed relative change since the previous refresh, e.g. ``0.707 (+0.14%)``;
when the baseline is zero or the change exceeds +/-100% it falls back to the
absolute delta, e.g. ``0.707 (+0.001)``.

The dataset
~~~~~~~~~~~

``<dataset>`` (and the dataset after ``--statistics-residual``) can be given in **two ways**:

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
    (``--statistics``, ``--statistics-tendencies``, ``--parallel``, ``--output``,
    ...) are **always** CLI flags and must never be put inside the JSON. Mixing
    ``key=value`` options with a JSON dataset is rejected — put every
    ``open_dataset`` option inside the JSON in that case.

If neither ``--statistics`` nor ``--statistics-tendencies`` is given, plain statistics are
computed by default. NaNs are ignored on a per-variable basis by default.

Options
-------

``--statistics``
    Compute the plain statistics (mean, minimum, maximum, stdev).

``--statistics-tendencies DELTA``
    Compute statistics of the tendencies ``value(t) - value(t - delta)`` for a
    single delta (e.g. ``6h``). The delta must be a whole multiple of the dataset
    frequency.

``--statistics-residual <dataset-2> [key=value ...]``
    Compute statistics of ``dataset - dataset-2``. The two datasets must share a
    length, variable list and field shape after their respective ``open_dataset``
    options are applied — use regrid/select options so that two datasets at
    different resolutions become comparable. ``<dataset-2>`` can be a name with
    ``key=value`` options or a single JSON ``open_dataset`` config.

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
    absolute and relative differences. Not applicable to ``--statistics-residual``.

``--output FILE.json``
    Write the results (and any ``--compare`` differences) to this JSON file. NaNs
    are written as ``null``. Results are always written: without ``--output`` the
    default path is ``<dataset-name>.statistics.json`` in the current directory.

``--overwrite``
    Replace the output file if it already exists. Without it the command fails
    immediately (before computing anything) when the output file is present.

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

    anemoi-datasets compute my-dataset --statistics-tendencies 6h --parallel 8 --output tend.json

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

Statistics of the difference between a high-resolution dataset regridded to a
low-resolution grid and the native low-resolution dataset:

.. code-block:: bash

    anemoi-datasets compute hi-res grid=o96 --statistics-residual lo-res

Resume a long run that was interrupted:

.. code-block:: bash

    anemoi-datasets compute my-dataset --statistics --parallel 16 --resume

Notes
-----

- The computation reads the dataset *as opened*, so any ``open_dataset`` option
  (selection, sub-area, rescaling, ...) is reflected in the recomputed statistics.
- ``--compare`` reads the dataset's stored statistics for the *full* opened
  dataset; if you restrict the period with ``start``/``end`` (or ``--sample-dates``)
  the recomputed values will legitimately differ from the stored ones.
- Trajectory datasets expose two frequencies and therefore support statistics
  but not ``--statistics-tendencies``.
