.. _complement-step:

##############################################
 Combining cutout with complementing datasets
##############################################

Here we explain how to combine a cutout with a complementing dataset.

****************************
 Interpolate to cutout grid
****************************

In this case, we will use the a ``lam-dataset`` in a different grid that
contains just one variable and a ``global-dataset``. What we want to do
is to interpolate the ``global-dataset`` to the resulting dataset from
the cutout grid operation.

.. literalinclude:: code/cutout-complement1.py

or for the config file:

.. literalinclude:: code/cutout-complement1.yaml

The ``adjust`` option is in case the end or start dates do not exactly
match.
