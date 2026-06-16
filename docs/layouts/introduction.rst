.. _layouts-introduction:

#################
 Dataset layouts
#################

The `anemoi-datasets` package supports three types of data organisation
(called "layouts"), all of which are designed to efficiently store and
access large datasets.

   -  :ref:`layouts-gridded` datasets are typically model
      fields featuring a (possibly unstructured) spatial grid that is regular in time.
   -  :ref:`layouts-tabular` datasets are typically observations which
      are unstructured both in time and space.
   -  :ref:`layouts-trajectories` datasets store forecast fields indexed
      by a base date (model-run time) and a forecast step, with a 5-D
      on-disk array.
