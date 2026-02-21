.. _layouts-gridded:

#########
 Gridded
#########

.. note::

   This page describes what is specific to the gridded layout. For more
   general information on creating and using datasets, see
   :ref:`using-introduction` and :ref:`building-introduction`
   respectively.

.. _gridded-creating:

**********
 Creating
**********

To create a gridded dataset, the ``layout`` entry in the recipe must be
set to ``gridded``:

Please note that this is the default value, so setting it is optional.

.. code:: yaml

   output:
     layout: gridded

.. _gridded-using:

*******
 Using
*******

Some text here.

.. code:: python

   ds = open_dataset(dataset, frequency="6h")

   ds.dates
   ds.frequency
   ds.latitudes
   ds.longitudes
