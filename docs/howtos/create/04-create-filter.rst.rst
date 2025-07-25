.. _create-filter:

#################################
 Create a dataset using a filter
#################################

A ``filter`` is a software component that takes as input the output of a
source or another filter and can modify the fields and/or their
metadata. For example, typical filters are interpolations, renaming of
variables, etc. Filters are available as part of ``anemoi-transform``.
Depending on the exact transformation or processing to be applied, there
exist two flavours of filters: - `SingleFieldFilter
<https://github.com/ecmwf/anemoi/blob/main/anemoi/preprocess/filters/single_field_filter.py>`_
- a filter which operates on a single field at a time -
`MatchingFieldsFilter<https://anemoi.readthedocs.io/projects/transform/en/latest/filters/matching-filters.html>_`
- a filter which operates on a set of fields, grouped and matched by
metadata, at a time

Both inherit from the base ``Filter`` class so if other uses cases
appear in the future, the type of filters can be extended.

################
 Using a filter
################

In the example below we see a recipe to create a dataset from MARS data
in which we perform a rename transform to update ``tp`` to be named
``tp_era5``. To be able to use the transform we just define it as a
second step of the pipe, after gathering the data.

.. literalinclude:: yaml/recipe-filter1.yaml

######################
 Creating a new filer
######################

In order to create a new filter the recommendation is to define it under
the package ``anemoi-transform``. Available filters can be found in
``anemoi/transform/filters`` or running the command ``anemoi-transform
filters list``. A filter should have two main methods: -
forward_transform: function to apply the transform to the raw data -
backward_transform: function to reverse the transform and recover the
raw data

.. note::

   This is general rule, there are certain operations that can be
   reversed. In those cases it's okey to just implement the
   forward_transform.

########################
 Using multiple filters
########################

It's possible to stack multiple filters one after the other. Below you
can see an updated version of the dataset creation we had where we now
create a dataset and apply a rename filter and our newly defined
``VerticalVelocity`` filter.
