.. _create-with-filter:

#################################
 Create a dataset using a filter
#################################

A ``filter`` is a software component that takes as input the output of a
source or another filter and can modify the fields and/or their
metadata. For example, typical filters are interpolations, renaming of
variables, etc. Filters are available as part of ``anemoi-transform``.

****************
 Using a filter
****************

In the example below we see a recipe to create a dataset from MARS data
in which we perform a rename transform to update ``tp`` to be named
``tp_era5``. To be able to use the transform we just define it as a
second step of the pipe, after gathering the data.

.. literalinclude:: yaml/recipe-filter1.yaml

That recipe will generate the following dataset:

.. code:: bash

   Dataset Summary
   ===============

   📦 Path          : recipe1.zarr
   🔢 Format version: 0.30.0

   📅 Start      : 2020-12-12 00:00
   📅 End        : 2020-12-13 12:00
   ⏰ Frequency  : 12h
   🚫 Missing    : 0
   🌎 Resolution : 20.0
   🌎 Field shape: [9, 18]

   📐 Shape      : 4 × 9 × 1 × 162 (22.8 KiB)
   💽 Size       : 40.7 KiB (40.7 KiB)
   📁 Files      : 48

      Index │ Variable │         Min │         Max │         Mean │       Stdev
      ──────┼──────────┼─────────────┼─────────────┼──────────────┼────────────
         0 │ 2t       │     226.496 │     309.946 │       278.03 │     19.2561
         1 │ cp       │           0 │  0.00739765 │   0.00014582 │ 0.000527194
         2 │ q_100    │ 1.38935e-06 │ 4.20381e-06 │  2.68779e-06 │ 5.59043e-07
         3 │ q_50     │ 1.26881e-06 │ 3.20919e-06 │  2.74525e-06 │ 4.35595e-07
         4 │ t_100    │     189.787 │     226.929 │      207.052 │     9.26841
         5 │ t_50     │      189.14 │      236.51 │       212.79 │      9.5502
         6 │ tp_era5  │           0 │  0.00823116 │  0.000326814 │  0.00078008
         7 │ w_100    │  -0.0114685 │   0.0129402 │ -0.000355278 │  0.00448272
         8 │ w_50     │ -0.00815806 │   0.0126816 │ -0.000267674 │  0.00331866
      ──────┴──────────┴─────────────┴─────────────┴──────────────┴────────────
   🔋 Dataset ready, last update 1 minute ago.
   📊 Statistics ready.

***********************
 Creating a new filter
***********************

In order to create a new filter the recommendation is to define it under
the package ``anemoi-transform``. Available filters can be found in
``anemoi/transform/filters`` or running the command ``anemoi-transform
filters list``. For details about how to create a filter please refer to
the `anemoi-transform
<https://anemoi.readthedocs.io/projects/transform/en/latest/>`_
documentation.

************************
 Using multiple filters
************************

It's possible to stack multiple filters one after the other. Below you
can see an updated version of the dataset creation we had where we now
create a dataset and apply a rename filter and our newly defined
``VerticalVelocity`` filter.

.. literalinclude:: yaml/recipe-filter2.yaml

That recipe will generate the following dataset:

.. code:: bash

   Dataset Summary
   ===============


   📦 Path          : recipe2.zarr
   🔢 Format version: 0.30.0

   📅 Start      : 2020-12-12 00:00
   📅 End        : 2020-12-13 12:00
   ⏰ Frequency  : 12h
   🚫 Missing    : 0
   🌎 Resolution : 20.0
   🌎 Field shape: [9, 18]

   📐 Shape      : 4 × 9 × 1 × 162 (22.8 KiB)
   💽 Size       : 41.1 KiB (41.1 KiB)
   📁 Files      : 48

      Index │ Variable │         Min │         Max │        Mean │       Stdev
      ──────┼──────────┼─────────────┼─────────────┼─────────────┼────────────
         0 │ 2t       │     226.496 │     309.946 │      278.03 │     19.2561
         1 │ cp       │           0 │  0.00739765 │  0.00014582 │ 0.000527194
         2 │ q_100    │ 1.38935e-06 │ 4.20381e-06 │ 2.68779e-06 │ 5.59043e-07
         3 │ q_50     │ 1.26881e-06 │ 3.20919e-06 │ 2.74525e-06 │ 4.35595e-07
         4 │ t_100    │     189.787 │     226.929 │     207.052 │     9.26841
         5 │ t_50     │      189.14 │      236.51 │      212.79 │      9.5502
         6 │ tp_era5  │           0 │  0.00823116 │ 0.000326814 │  0.00078008
         7 │ wz_100   │ -0.00798191 │  0.00721723 │ 0.000224189 │  0.00277693
         8 │ wz_50    │    -0.01549 │   0.0103844 │ 0.000341309 │  0.00417065
      ──────┴──────────┴─────────────┴─────────────┴─────────────┴────────────
   🔋 Dataset ready, last update 11 seconds ago.
   📊 Statistics ready.
