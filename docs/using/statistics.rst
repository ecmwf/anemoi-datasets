.. _using-statistics:

############
 Statistics
############

When combining datasets, the statistics are not recomputed. Instead, the
statistics of the first dataset encountered are returned by the
``statistics`` property.

You can change that behaviour by using the `statistics` option to select
a specific dataset from which to get the statistics:

.. literalinclude:: code/statistics_select_.py
