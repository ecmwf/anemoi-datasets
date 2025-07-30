.. _create-with-filter:

#################################
 Create a dataset using a filter
#################################

A ``filter`` is a software component that takes as input the output of a
source or another filter and can modify the fields and/or their
metadata. For example, typical filters are interpolations, renaming of
variables, etc. Filters are available as part of ``anemoi-transform``.

Both inherit from the base ``Filter`` class so if other uses cases
appear in the future, the type of filters can be extended.

****************
 Using a filter
****************

In the example below we see a recipe to create a dataset from MARS data
in which we perform a rename transform to update ``tp`` to be named
``tp_era5``. To be able to use the transform we just define it as a
second step of the pipe, after gathering the data.

.. literalinclude:: yaml/recipe-filter1.yaml

***********************
 Creating a new filter
***********************

In order to create a new filter the recommendation is to define it under
the package ``anemoi-transform``. Available filters can be found in
``anemoi/transform/filters`` or running the command ``anemoi-transform
filters list``. For details about how to create a filter please refer to
the `anemoi-transform
<https://anemoi.readthedocs.io/projects/transform/en/latest/>`_
documentation.

.. note::

   This is a general rule, there are certain operations that can't be
   reversed. In those cases it's okey to just implement the
   forward_transform.

************************
 Using multiple filters
************************

It's possible to stack multiple filters one after the other. Below you
can see an updated version of the dataset creation we had where we now
create a dataset and apply a rename filter and our newly defined
``VerticalVelocity`` filter.

.. literalinclude:: yaml/recipe-filter2.yaml
