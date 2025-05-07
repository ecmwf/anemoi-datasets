.. _create-filter:


#################################
 Create a dataset using an filter
#################################

When creating a dataset, it might be useful to perform certain operations to the raw data such as unit conversion, reprojections, filtering of potential outliers, etc.
In anemoi-dataset such operation can be performed using 'Filters'. A filter is a software component that takes as input the output of a source or another filter and can modify the fields and/or their metadata.
For example, typical filters are interpolations, renaming of variables, etc. 

****************
 Using a filter
****************

In the example below we see a recipe to create a dataset from MARS data in which we perform
a rename transform to update 'tp' to be named 'tp_era5'. To be able to use the transform we just define it
as a second step of the pipe, after gathering the data. 

.. literalinclude:: yaml/recipe-filter1.yaml

**********************
 Creating a new filer
**********************

In order to create a new filter the recommendation is to define it under the package anemoi-transform. 
Available filters can be found in ``anemoi/transform/filters``. We have a base class that should be used to define 
any filter, called  ``Filter`` in  ``src/anemoi/transform/filter.py``. This is a placeholder class intended to be used to expand it
with logic specific for each filter. For the majority of the use cases what we need is a filter that convert part of the fields variables, 
required a match between the updated variable and its metadata. For those use cases, we have defined a  ``MatchingFieldsFilter``, which can 
be found in  ``src/anemoi/transform/filters/matching.py``. A filter should have two main methods:
- forward_transform: function to apply the transform to the raw data
- backward_transform: function to reverse the transform and recover the raw data

In order to be able to use a filter we need to register it. We do that using a decorator called  ``@filter_registry``. To look at these pieces together, let's look at an example where
we have a field with 5 variables and out of those we want to convert wind speed from m/s to vertical to vertical wind speed expressed in Pa/s using the hydrostatic hypothesis,
and back. 

.. literalinclude:: code/filter.py


In the example below we can see how we need to make sure that the  ``@matching`` decorator is consistent with the inputs defined both for the forward and backward transform.  

.. warning::
    Please note: there are additional filters that can be found under  ``anemoi/datasets/create/filters``. These filters have a legacy design pattern and we are in the process of migrating and updating those to anemoi-transform. 

************************
 Using multiple filters
***********************

It's possible to stack multiple filter one after the other. Below you can see an updated version
of the dataset creation we had where we now create a dataset and apply a rename filter and our newly defined ``VerticalVelocity`` filter.
