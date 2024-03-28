.. _selecting-statistics:

############
 Statistics
############

When combining dataset, the statistics are not recomputed;
   it is the statistics of first dataset encounter that are returned by
   the ``statistics`` method.

You can change that behaviour by using the `statistics` option to select
a specific dataset statistics:

.. literalinclude:: code/statistics_.py
   :language: python
