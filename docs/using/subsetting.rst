.. _subsetting-datasets:

#####################
 Subsetting datasets
#####################

Subsetting is the action of filtering the dataset by it's first
dimension (dates).

.. _start:

*******
 start
*******

This option lets you subset the dataset by time. You can pass a date or
a string:

.. literalinclude:: code/start_.py

.. _end:

*****
 end
*****

As for the start option, you can pass a date or a string:

.. literalinclude:: code/end_.py
   :language: python

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

.. literalinclude:: code/frequency1_.py
   :language: python

The new frequency must be a multiple of the original frequency.

To artificially increase the frequency, you can use the
``interpolate_frequency`` option. This will create new dates in the
dataset by linearly interpolating the data values between the original
dates.

.. literalinclude:: code/frequency2_.py
   :language: python
