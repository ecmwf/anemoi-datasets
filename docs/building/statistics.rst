.. _gathering_statistics:

######################
 Gathering statistics
######################

*Anemoi* will collect statistics about each variables in the dataset as
it is created. These statistics are intended to be used to normalise the
data during training.

By defaults, the statistics are not computed on the whole dataset, but
on a subset of dates. The subset is defined using the following
algorithm:

   -  If the dataset covers more than 20 years, the last 3 years are
      excluded.
   -  If the dataset covers more than 10 years, the last 2 years are
      excluded.
   -  If the dataset covers more than 5 years, the last year is
      excluded.
   -  Otherwise, 80% of the dataset is used.

You can override this behaviour by setting the `statistics_dates`
parameter.

.. code:: yaml

   output:
       statistics_start: 2000
       statistics_end: 2020

..
   .. todo:: List the statistics that are computed
