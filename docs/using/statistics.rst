.. _selecting-statistics:

############
 Statistics
############

When combining datasets, the statistics are not recomputed. Instead, the
statistics of the first dataset encountered are returned by the
``statistics`` property.

You can change that behaviour by using the `statistics` option to select
a specific dataset from which to get the statistics:

.. code:: python

   ds = open_dataset(dataset, statistics=other_dataset)

   # Will return the statistics of "other_dataset"

   print(ds.statistics)
