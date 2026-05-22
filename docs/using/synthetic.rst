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
         "grid": {"bbox": [60, -10, 30, 20], "resolution": 0.25},
         "variables": ["2t", "msl", "z_500"],
         "dates": {"start": "2020-01-01", "end": "2020-01-31", "frequency": "6h"},
         "values": {
            "default": {"mode": "random", "mean": 0.0, "std": 1.0},
            "2t": {"mode": "constant", "value": 273.15},
         },
      }
   )

The synthetic dataset is fully described by the ``synthetic`` dictionary and does not
combine with other ``open_dataset`` keywords --- express the full extent inside
the dictionary. It accepts the following keys.

``grid`` (required)
   The spatial extent. Exactly one of:

   -  ``{"bbox": [north, west, south, east], "resolution": 0.25}`` --- a regular
      latitude/longitude mesh. ``resolution`` may be a scalar or ``[dlat, dlon]``.
   -  ``{"named": "o96"}`` --- a named grid resolved through
      ``anemoi.transform.grids``.
   -  ``{"icon": {"path": "grid.nc", "refinement_level_c": 7}}`` --- an ICON grid
      file.
   -  ``{"unstructured": {"latitudes": [...], "longitudes": [...]}}`` --- explicit
      coordinate arrays (or a path to an ``.npz`` file).

``variables`` (required)
   Either an explicit list of names, or an integer ``N`` to auto-generate
   ``var_00 ... var_{N-1}``.

``dates`` (required)
   The temporal extent, as a dict of ``start``, ``end`` and ``frequency`` ---
   mirroring the ``dates`` block of a dataset-building recipe. ``start`` and
   ``end`` are ISO date/datetime strings; ``frequency`` is an anemoi frequency
   string such as ``"6h"`` or ``"1d"``.

``values`` (optional)
   Per-variable value specs. The ``default`` entry covers any variable not named
   explicitly. Each spec has a ``mode``:

   -  ``{"mode": "constant", "value": 273.15}``
   -  ``{"mode": "random", "mean": 0.0, "std": 1.0}`` --- seeded Gaussian noise,
      reproducible for a given ``seed``.
   -  ``{"mode": "index"}`` --- each value encodes its own
      ``(date, variable, ensemble, gridpoint)`` position, useful for verifying a
      pipeline does not shuffle or misalign data.

   If ``values`` is omitted, every variable defaults to ``random`` with mean 0
   and standard deviation 1.

``ensemble`` (optional, default ``1``)
   The number of ensemble members.

``seed`` (optional, default ``0``)
   Seed for the ``random`` value mode.

``dtype`` (optional, default ``float32``)
   The dtype of the generated data.

``resolution`` (optional)
   Overrides the resolution string reported by ``ds.resolution`` and the
   dataset metadata. By default it is derived from the ``grid`` --- the
   ``bbox`` spacing or the ``named`` grid name --- and is ``"unknown"`` for
   ``icon`` and ``unstructured`` grids.
