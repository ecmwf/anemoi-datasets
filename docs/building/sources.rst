.. _sources:

#########
 Sources
#########

The source is a software component that given a list of dates and
variables will return the corresponding fields.

A `source` is responsible for reading data from the source and
converting it to a set of fields. A `source` is also responsible for
handling the metadata of the data, such as the variables names, and
more.

A example of source is ECMWF’s MARS archive, a collection of GRIB or
NetCDF files, etc.

The following `sources` are currently available:

.. toctree::
   :maxdepth: 1

   sources/mars
   sources/grib
   sources/netcdf
   sources/xarray
   sources/opendap
   sources/forcings
   sources/accumulations
   sources/recentre
