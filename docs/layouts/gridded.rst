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

See :ref:`building-gridded-cli` for a step-by-step guide to building a
gridded dataset with the command line tool.

.. _gridded-using:

*******
 Using
*******

In addition to the parameters described in :ref:`using-introduction`,
you can access the following attributes of a gridded dataset:

.. code:: python

   ds.latitudes # Latitudes of the grid points

   ds.longitudes # Longitudes of the grid points
