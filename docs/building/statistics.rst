.. _gathering_statistics:

######################
 Gathering statistics
######################

*Anemoi* will collect statistics about each variable and its tendencies
in the dataset as it is created. These statistics are intended to be
used to normalise the data during training.

The statistics are stored in the :doc:`statistics attribute
<../using/statistics>` of the dataset. The computed statistics include
`minimum, maximum, mean, standard deviation`.

Statistics about the tendencies are returned by calling the
statistics_tendencies() method.

For example:

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset("aifs-ea-an-oper-0001-mars-o48-2020-2021-6h-v1.zarr",
                     select="2t")

   ds.statistics

   {'mean': array([287.82987893]), 'stdev': array([16.139444]),
   'maximum': array([325.78491211]), 'minimum': array([195.36195374])}

   ds.statistics_tendencies()

   {'mean': array([0.00164934]), 'stdev': array([2.80699639]),
   'maximum': array([27.10545349]), 'minimum': array([-29.63409424])}

************************
 Statistics dates range
************************

By default, the statistics are not computed on the whole dataset, but on
a subset of dates. This is usually done to avoid any data leakage from
the validation and test sets to the training set.

The dates subset used to compute the statistics is defined using the
following algorithm:

   -  If the dataset covers 20 years or more, the last 3 years are
      excluded.
   -  If the dataset covers 10 years or more, the last year is excluded.
   -  Otherwise, 80% of the dataset is used.

You can override this behaviour by setting either the `start` parameter
or the `end` parameter in the `statistics` config.

Example configuration gathering statistics from 2000 to 2020:

.. code:: yaml

   statistics:
       start: 2000
       end: 2020

Example configuration gathering statistics from the beginning of the
dataset period to 2020:

.. code:: yaml

   statistics:
       end: 2020

Example configuration gathering statistics using only 2020 data:

.. code:: yaml

   statistics:
       start: 2020
       end: 2020

**************************
 Data with missing values
**************************

If the dataset contains missing values (known as `NaNs`), an error will
be raised when trying to compute the statistics. To allow `NaNs` in the
dataset, you can set the `allow_nans` as described :doc:`here
<handling-missing-values>`.
