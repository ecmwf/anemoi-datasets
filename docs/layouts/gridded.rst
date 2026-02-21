.. _layouts-gridded:

#########
 Gridded
#########

.. note::

   This page describe what is specific to the gridded layout. For more
   general information creating and using datasets, see
   :ref:`using-introduction` and :ref:`building-introduction`
   repectively.

.. _gridded-creating:

**********
 Creating
**********

To create a gridded dataset, the ``layout`` entry in the recipe must be
set to ``gridded``:

Please not that this is the default value, so setting it is optional.

.. code:: yaml

   output:
     layout: gridded

.. _gridded-using:

*******
 Using
*******

Soem text here.

.. code:: python

   ds = open_dataset(dataset, frequency="6h")

   ds.dates
   ds.frequency
   ds.latitudes
   ds.longitudes
