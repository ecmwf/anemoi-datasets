.. _synthetic-datasets:

#################
 Synthetic datasets
#################

A *synthetic dataset* is an in-memory synthetic dataset. It builds no Zarr store on
disk and is intended for testing and prototyping training and inference
pipelines. Open one by passing a ``synthetic`` dictionary to ``open_dataset``:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
      synthetic={
         "geography": {"bbox": [60, -10, 30, 20], "resolution": 0.25},
         "dates": {"start": "2020-01-01", "end": "2020-01-31", "frequency": "6h"},
         "layout": "gridded",
         "variables": [
            {"name": "2t", "values": {"constant": 273.15}},
            "msl",
            "insolation",
         ],
      }
   )

The dataset itself is described entirely by the ``synthetic`` dictionary, which
accepts the following keys. ``synthetic`` is a drop-in replacement for ``dataset``:
the usual transform keywords (``select``, ``start``, ``end``, ``rename``, ...) may
be passed alongside it and apply to the result, just as for any other dataset.

``geography`` (required)
   The spatial extent. Exactly one of:

   -  ``{"bbox": [north, west, south, east], "resolution": 0.25}`` --- a regular
      latitude/longitude mesh. ``resolution`` may be a scalar or ``[dlat, dlon]``.
   -  ``{"named": "o96"}`` --- a named grid resolved through
      ``anemoi.transform.grids``.
   -  ``{"icon": {"path": "grid.nc", "refinement_level_c": 7}}`` --- an ICON grid
      file.
   -  ``{"unstructured": {"latitudes": [...], "longitudes": [...]}}`` --- explicit
      coordinate arrays (or a path to an ``.npz`` file).

``dates`` (required)
   The temporal extent, as a dict of ``start``, ``end`` and ``frequency`` ---
   mirroring the ``dates`` block of a dataset-building recipe. ``start`` and
   ``end`` are ISO date/datetime strings; ``frequency`` is an anemoi frequency
   string such as ``"6h"`` or ``"1d"``.

``variables`` (required)
   A list whose entries are either a **string** (the variable name, using the
   dataset-level default value generator) or a **dict** with a mandatory ``name``
   and any of these optional keys:

   -  ``values`` --- how the variable's values are generated (see below). Omit to
      use the dataset-level default.
   -  ``metadata`` --- a per-variable metadata block, surfaced on the dataset's
      ``variables_metadata``.
   -  ``statistics`` --- override the analytic statistics, e.g.
      ``{"mean": 0.0, "stdev": 1.0}``. Unspecified keys keep their analytic value.
   -  ``tendencies_statistics`` --- override the analytic tendency statistics.

   A variable whose name is a **computed forcing** (``insolation``,
   ``cos_latitude``, ``sin_julian_day``, ``cos_local_time``, ...) is generated from
   the grid and dates through earthkit's forcings source. It must be given as a
   string (or a dict without a ``values`` block) --- it owns its own value
   generation.

``layout`` (required)
   One of ``gridded``, ``tabular`` or ``trajectories``. Only ``gridded`` is
   implemented; the other two raise ``NotImplementedError``.

``values`` (optional)
   The dataset-level **default** value generator, used for any variable that does
   not carry its own ``values``. A value spec is one of:

   -  ``{"constant": 273.15}`` --- a fixed value (given directly).
   -  ``{"random": {"mean": 0.0, "std": 1.0}}`` --- seeded Gaussian noise,
      reproducible for a given ``seed``. Parameters are optional.
   -  ``273.15`` --- a bare scalar, shorthand for ``{"constant": 273.15}``.
   -  ``"random"`` --- a bare generator name, using its default parameters.

   If ``values`` is omitted, every non-forcing variable defaults to ``random`` with
   mean 0 and standard deviation 1.

``ensembles`` (optional, default ``1``)
   The number of ensemble members.

``seed`` (optional, default ``0``)
   Seed for the ``random`` value generator.

``dtype`` (optional, default ``float32``)
   The dtype of the generated data.

``resolution`` (optional)
   Overrides the resolution string reported by ``ds.resolution`` and the
   dataset metadata. By default it is derived from the ``geography`` --- the
   ``bbox`` spacing or the ``named`` grid name --- and is ``"unknown"`` for
   ``icon`` and ``unstructured`` grids.
