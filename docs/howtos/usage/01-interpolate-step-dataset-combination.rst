.. _interpolate-step:

######################################################
 Combine datasets with different timestep frequencies
######################################################

Here we explain how to combine two existing datasets with different
timestep frequencies. In this example we consider two datasets :
``dataset1`` with a te,mporal frequency of 3h and ``dataset2`` with a
temporal frequency of 24h. The goal is to combine the two datasets into
a single dataset with a temporal frequency of 3h or 24h. We consider two
cases, in case one we would like to bring the larger timestep dataset to
the smaller timestep dataset, in case two we would like to bring the
smaller timestep dataset to the larger timestep dataset.

*********************************
 Interpolate to higher frequency
*********************************

In this case we will use the ``interpolate_frequency`` option to bring
``dataset2`` to the 3h timestep of dataset1.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
      dataset={
         "join": [
               {
                  "dataset": "dataset1",
                  "frequency": "3h",
               },
               {
                  "dataset": "dataset2",
                  "interpolate_frequency": "3h",
               },
         ],
         "adjust": "dates",
      },
      start="2004-01-01",
      end="2023-01-01",
      )

or in the config file

.. literalinclude:: yaml/interpolate_frequencies.zaml
   :language: yaml

The ``adjust`` option is in case the end or start dates do not exactly
match.

***************************
 Sample to lower frequency
***************************

This case is straightforward, we will can just specify the required 24h
frequency for datset1.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
      dataset={
         "join": [
               {
                  "dataset": "dataset1",
                  "frequency": "24h",
               },
               {
                  "dataset": "dataset2",
                  "frequency": "24h",
               },
         ],
         "adjust": "dates",
      },
      start="2004-01-01",
      end="2023-01-01",
      )

of for the config file

.. literalinclude:: yaml/sample.yaml
   :language: yaml
