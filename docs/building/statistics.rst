.. _gathering_statistics:

Gathering statistics
====================

*Anemoi* will collect statistics about each variables in the dataset as it is created.
These statistics are intended to be used to normalise the data during training.

The statistics are stored in the `statistics` attribute of the dataset. The computed
statistics include:

- Minimum
- Maximum
- Mean
- Standard deviation

By defaults, the statistics are not computed on the whole dataset, but on a subset of
dates. The subset is defined using the following algorithm:

    - If the dataset covers 20 years or more, the last 3 years are excluded.
    - If the dataset covers 10 years or more, the last year is excluded.
    - Otherwise, 80% of the dataset is used.

You can override this behaviour by setting either the `start` parameter or the `end`
parameters in the `statistics` config.

Example configuration gathering statistics from 2000 to 2020 :

.. code-block:: yaml

    statistics:
        start: 2000
        end: 2020

Example configuration gathering statistics from the beginning of the dataset period to
2020 :

.. code-block:: yaml

    statistics:
        end: 2020

Example configuration gathering statistics using only 2020 data :

.. code-block:: yaml

    statistics:
        start: 2020
        end: 2020
