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

.. literalinclude:: code/subsetting_start_.py

.. _end:

*****
 end
*****

As for the start option, you can pass a date or a string:

.. literalinclude:: code/subsetting_end_.py

The following are equivalent ways of describing ``start`` or ``end``:

-  ``2020`` and ``"2020"``
-  ``202306``, ``"202306"`` and ``"2023-06"``
-  ``20200301``, ``"20200301"`` and ``"2020-03-01"``

Note that the ``start="2020"`` is equivalent to ``start="2020-01-01"``
while ``end="2020"`` is equivalent to ``end="2020-12-31"``.

Note also how the ``frequency`` of the dataset will change how the
``end`` option is interpreted:

-  ``end="2020"`` with a ``frequency`` of one hour is equivalent to
   ``end="2020-12-31 23:00:00"``
-  ``end="2020"`` with a ``frequency`` of 6 hours is equivalent to
   ``end="2020-12-31 18:00:00"``

.. _frequency:

***********
 frequency
***********

You can change the frequency of the dataset by passing a string with:

.. literalinclude:: code/subsetting_frequency_.py

The new frequency must be a multiple of the original frequency.

To artificially increase the frequency, you can use the
``interpolate_frequency`` option. This will create new dates in the
dataset by linearly interpolating the data values between the original
dates.

.. literalinclude:: code/subsetting_interpolate_frequency_.py

.. _extend:

********
 extend
********

You can extend the date range of a dataset backwards and/or forwards by
using the ``extend_start`` and ``extend_end`` options. The new dates are
added at the dataset's ``frequency`` and are marked as :ref:`missing
<selecting-missing>`, so accessing them will raise a ``MissingDateError``
unless you combine them with a :ref:`fill_missing_dates
<selecting-missing>` method.

.. code:: python

   ds = open_dataset(dataset, extend_start="2019-01-01", extend_end="2021-12-31")

The ``extend_start`` date must be before (or equal to) the first date of
the dataset, and the ``extend_end`` date must be after (or equal to) the
last date. Either option can be omitted to extend in only one direction:

.. code:: python

   # Extend only backwards
   ds = open_dataset(dataset, extend_start="2019-01-01")

   # Extend only forwards
   ds = open_dataset(dataset, extend_end="2021-12-31")

As with the :ref:`start` and :ref:`end` options, you can pass a partial
date, and it will be expanded taking the dataset's ``frequency`` into
account. The ``extend_start`` is expanded to the first time step of the
period, while ``extend_end`` is expanded to the last time step:

-  ``extend_start="2019"`` is equivalent to ``extend_start="2019-01-01
   00:00:00"``
-  ``extend_end="2021"`` is equivalent to ``extend_end="2021-12-31
   18:00:00"`` for a 6-hourly dataset
-  ``extend_end="2021-06"`` is equivalent to ``extend_end="2021-06-30
   18:00:00"`` for a 6-hourly dataset

Unlike ``start`` and ``end``, the added dates do not need to already
exist in the dataset, so no reference dates are used when expanding.

This is typically combined with a fill method so that the added dates
hold artificial values instead of raising an error:

.. code:: python

   ds = open_dataset(
       dataset,
       extend_start="2019-01-01",
       extend_end="2021-12-31",
       fill_missing_dates="nans",
   )

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

.. literalinclude:: code/subsetting_traj_step_.py

Base-date axis
==============

``base_start`` and ``base_end`` filter the base-date axis directly,
without the envelope logic used by ``start`` / ``end``:

.. literalinclude:: code/subsetting_traj_base_.py
