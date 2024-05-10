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

   -  If the dataset covers 20 years or more, the last 3 years are
      excluded.
   -  If the dataset covers 10 years or more, the last year is excluded.
   -  Otherwise, 80% of the dataset is used.

You can override this behaviour by setting the `start` or `end`
parameters in the `statistics` config.

.. code:: yaml

   statistics:
       start: 2000
       end: 2020

..
   TODO: List the statistics that are computed
