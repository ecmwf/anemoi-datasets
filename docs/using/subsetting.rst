.. _subsetting-datasets:

#####################
 Subsetting datasets
#####################

*******
 start
*******

This option let you subset the dataset by time. You can pass a date or a

.. code:: python

   open_dataset(dataset, start=1980)

*****
 end
*****

As for the start option, you can pass a date or a string:

.. code:: python

   open_dataset(dataset, end="2020-12-31")

The following are equivalent way of describing ``start`` or ``end``:

-  ``2020`` and ``"2020"``
-  ``202306``, ``"202306"`` and ``"2023-06"``
-  ``20200301``, ``"20200301"`` and ``"2020-03-01"``

***********
 frequency
***********

You can change the frequency of the dataset by passing a string with the

.. code:: python

   ds = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
      frequency="6h")
