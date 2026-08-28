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
