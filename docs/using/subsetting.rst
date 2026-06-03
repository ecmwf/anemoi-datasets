.. _subsetting-datasets:

#####################
 Subsetting datasets
#####################

Subsetting is the action of filtering the dataset by its first dimension
(dates).

.. _start:

*******
 start
*******

This option lets you subset the dataset by time. You can pass a date or
a string:

.. code:: python

   open_dataset(dataset, start=1980)

.. _end:

*****
 end
*****

As for the start option, you can pass a date or a string:

.. code:: python

   open_dataset(dataset, end="2020-12-31")

The following are equivalent ways of describing ``start`` or ``end``:

-  ``2020`` and ``"2020"``
-  ``202306``, ``"202306"`` and ``"2023-06"``
-  ``20200301``, ``"20200301"`` and ``"2020-03-01"``

Note that the ``start="2020"`` is equivalent to ``start="2020-01-01"``
while ``end="2020"`` is equivalent to ``end="2020-12-31"``.

Note also how the ``frequency`` of the dataset will change how the
``end`` option is interpreted: - ``end="2020"`` with a ``frequency`` of
one hour is equivalent to ``end="2020-12-31 23:00:00"`` - ``end="2020"``
with a ``frequency`` of 6 hours is equivalent to ``end="2020-12-31
18:00:00"``

.. _frequency:

***********
 frequency
***********

You can change the frequency of the dataset by passing a string with:

.. code:: python

   ds = open_dataset(dataset, frequency="6h")

The new frequency must be a multiple of the original frequency.

To artificially increase the frequency, you can use the
``interpolate_frequency`` option. This will create new dates in the
dataset by linearly interpolating the data values between the original
dates.

.. code:: python

   ds = open_dataset(dataset, interpolate_frequency="10m")

.. _subsetting-trajectories:

**************************
 Trajectories-only options
**************************

For :ref:`trajectory datasets <layouts-trajectories>`, ``open_dataset``
accepts a few extra keyword arguments to subset the step axis and the
base-date axis independently of the ``start`` / ``end`` / ``frequency``
options (which continue to work with envelope semantics: a base date is
kept iff ``[base + step_start, base + step_end] ⊂ [start, end]``).

Step axis
=========

.. code:: python

   # Select a single forecast step; returns a 4-D view
   # (base_dates, variables, ensembles, cells) — shape-compatible
   # with a gridded dataset at that lead time.
   ds_t6 = open_dataset("traj.zarr", step=6)

   # Select a list of steps; keeps the 5-D shape, narrows the step axis.
   ds_subset = open_dataset("traj.zarr", steps=[6, 12, 18])

   # Step range form (all three are optional).
   ds_range = open_dataset("traj.zarr",
                           step_start=6, step_end=24, step_frequency="6h")

Base-date axis
==============

``base_start`` and ``base_end`` filter the base-date axis directly,
without the envelope logic used by ``start`` / ``end``:

.. code:: python

   ds_jan = open_dataset("traj.zarr",
                         base_start="2021-01-01",
                         base_end="2021-01-31")
