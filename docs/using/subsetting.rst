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

This option let you subset the dataset by time. You can pass a date or a

.. literalinclude:: code/start_.py

.. _end:

*****
 end
*****

As for the start option, you can pass a date or a string:

.. literalinclude:: code/end_.py
   :language: python

The following are equivalent way of describing ``start`` or ``end``:

-  ``2020`` and ``"2020"``
-  ``202306``, ``"202306"`` and ``"2023-06"``
-  ``20200301``, ``"20200301"`` and ``"2020-03-01"``

.. _frequency:

***********
 frequency
***********

You can change the frequency of the dataset by passing a string with the

.. literalinclude:: code/frequency_.py
   :language: python
