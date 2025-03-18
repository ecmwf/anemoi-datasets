.. _selecting-statistics:

############
 Statistics
############

When combining dataset, the statistics are not recomputed. Instead, the
the statistics of first dataset encounter that are returned by the
``statistics`` property.

You can change that behaviour by using the `statistics` option to select
a specific dataset from which to get the statistics:

.. literalinclude:: code/statistics_.py
   :language: python
