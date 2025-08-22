.. _create-accumulations:

##########################################
 Create a dataset with accumulated fields
##########################################


Many fields come as accumulations over time, e.g `tp` (total precipitations), `sd` (now depth) or `ssr` (surface shortwave radiation).
Given one dataset, one may want to accumulate some of its fields on specific periods of time.

This depends on the data's native format. For an accumulated field (say `tp` for simplicity), one needs to know:

- the `accumulation_period` over which to accumulate (e.g 6h).

- the desired `validity_time` at which accumulation stops and for which the value is valid.

- the `data_accumulation_period`, that is the duration over which the data is already accumulated.

The resulting field is then "`tp` accumulated over `accumulation_period` hours up to `validity_time`".
In a common case, dataset features, e.g., 1h-accumulated `tp` at a 1 hour frequency, and each raw file features `tp` as accumulated over the *last* hour.
So having 6h-accumulated `tp` consists in taking all 6 files before (and including) `validity_time` and summing fields in them.

The resulting accumulated field can be treated as a normal anemoi `source` in recipes (e.g, filters can be applied to the source).

Note that depending on how your native dataset is built (e.g, your native files feature the accumulation on the *next* hour), the calculation can be very different.
See $Subtleties below with the associated recipes.

***************************************
 Using accumulations in recipes : mars
***************************************

In the example below we see recipes to create accumulations from MARS data.
To keep older recipes working, there are two equivalent ways to do so.
in which we perform a rename transform to update ``tp`` to be named
``tp_era5``. To be able to use the transform we just define it as a
second step of the pipe, after gathering the data.

.. literalinclude:: yaml/recipe-accumulate-era.yaml

That recipe will generate the following dataset:

.. code:: bash

   📦 Path       : recipe-accumulate.zarr
   🔢 Format version: 0.30.0

   📅 Start      : 2021-01-10 18:00
   📅 End        : 2021-01-12 12:00
   ⏰ Frequency  : 6h
   🚫 Missing    : 0
   🌎 Resolution : 20.0
   🌎 Field shape: [162]

   📐 Shape      : 8 × 2 × 1 × 162 (10.1 KiB)
   💽 Size       : 23.2 KiB (23.2 KiB)
   📁 Files      : 52

      Index │ Variable │ Min │       Max │        Mean │      Stdev
      ──────┼──────────┼─────┼───────────┼─────────────┼───────────
         0  │ cp       │   0 │ 0.0110734 │ 0.000244731 │ 0.00103593
         1  │ tp       │   0 │ 0.0333021 │  0.00058075 │ 0.00210331
      ──────┴──────────┴─────┴───────────┴─────────────┴───────────
   🔋 Dataset ready, last update 26 seconds ago.
   📊 Statistics ready.

The "legacy" way to do is the following (syntax is only slightly different)

.. literalinclude:: yaml/recipe-accumulation-era.yaml

The resulting dataset is:

.. code:: bash
   📦 Path       : recipe-accumulation.zarr
   🔢 Format version: 0.30.0

   📅 Start      : 2021-01-10 18:00
   📅 End        : 2021-01-12 12:00
   ⏰ Frequency  : 6h
   🚫 Missing    : 0
   🌎 Resolution : 20.0
   🌎 Field shape: [9, 18]

   📐 Shape      : 8 × 2 × 1 × 162 (10.1 KiB)
   💽 Size       : 22.2 KiB (22.2 KiB)
   📁 Files      : 52

      Index │ Variable │ Min │       Max │        Mean │      Stdev
      ──────┼──────────┼─────┼───────────┼─────────────┼───────────
         0  │ cp       │   0 │ 0.0110734 │ 0.000244739 │ 0.00103593
         1  │ tp       │   0 │ 0.0333023 │ 0.000580769 │ 0.00210332
      ──────┴──────────┴─────┴───────────┴─────────────┴───────────
   🔋 Dataset ready, last update 3 minutes ago.
   📊 Statistics ready.

Note that statitics for the two datasets are equal up to `1e-6`, this is due to rounding errors that can accumulate.
Larger discrepancies are a sign something might be wrong.

*********************************************
 Using accumulations in recipes : grib files
*********************************************

If your data source is grib files, you can use a grib-index as a source.
First create a `grib-index
<https://anemoi.readthedocs.io/projects/datasets/en/latest/howtos/create/01-grib-data.html#using-an-index-file>`_

that creates a database to query fields. Then, say we want to accumulate 3h-data over 6h.

.. literalinclude::  yaml/recipe-accumulate-gribindex.yaml

Note that we also added a filter at the end of the recipe to rename `tp` to `tp_6h`.
The frequency of the dataset is `1h`, so the accumulation is a moving window.

************
 Subtleties
************

Some datasets (such as ERA5) feature
