.. _selecting-variables:

#####################
 Selecting variables
#####################

********
 select
********

.. code:: python

   # Select '2t' and 'tp' in that order

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
       select=["2t", "tp"],
   )

.. code:: python

   # Select '2t' and 'tp', but preserve the order in which they are in the dataset

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
       select={"2t", "tp"},
   )

******
 drop
******

You can also drop some variables:

.. code:: python

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
       drop=["10u", "10v"],
   )

*********
 reorder
*********

and reorder them:

... using a list

.. code:: python

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
       reorder=["2t", "msl", "sp", "10u", "10v"],
   )

... or using a dictionary

.. code:: python

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
       reorder={"2t": 0, "msl": 1, "sp": 2, "10u": 3, "10v": 4},
   )

********
 rename
********

You can also rename variables:

.. code:: python

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
       rename={"2t": "t2m"},
   )

This will be useful when your join datasets and do not want variables
from one dataset to override the ones from the other.
