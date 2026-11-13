.. _filters:

#########
 Filters
#########

Filters can be used to perform data transformations when creating a
dataset. See the :ref:`anemoi-transform
<anemoi-transform:list-of-filters>` documentation for a list of filters
and example configurations.

*******************************
 Filters for tabular datasets
*******************************

The following filter is provided by ``anemoi-datasets`` itself.

add_gridded_fields
==================

Adds columns with the values of gridded fields, retrieved from MARS, at
the grid point nearest to each observation. It is meant for fields that
are constant in time, such as the land-sea mask or the orography; if the
request returns several fields for a parameter, only the first one is
used. The new columns are named after the MARS ``param`` short names.
The fields are retrieved only once per process.

.. code:: yaml

   input:
     pipe:
       - netcdf-tabular:
           path: /path/to/observations_*.nc
       - add_gridded_fields:
           mars:
             class: ea
             expver: "0001"
             stream: oper
             type: an
             levtype: sfc
             param: [lsm, z, sdor, slor]
             grid: n320
             date: 2023-12-01
             time: "00:00"
