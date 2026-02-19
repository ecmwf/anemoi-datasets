.. _layouts-introduction:

#################
 Dataset layouts
#################

The `anemoi-datasets` package supports two types of data organisation
(called "layouts"), both of which are designed to efficiently store and
access large datasets:

*********
 Gridded
*********

Gridded data are typically meteorological fields. All the fields must be
defined on the same grid and avaliable at the same time frequencies.

*********
 Tabular
*********

Tabular data are typically observations. Each observation has its own
time and location. All observation shoul have the same set of variables,
which can be NaNs for some observations.
