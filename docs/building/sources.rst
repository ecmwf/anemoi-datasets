.. _sources:

#########
 Sources
#########

The source is a software component that, given a list of dates and
variables, will return the corresponding fields.

A `source` is responsible for reading data from the source and
converting it to a set of fields. A `source` is also responsible for
handling the metadata of the data, such as the variable names, and more.

An example of a source is ECMWF’s MARS archive, a collection of GRIB or
NetCDF files, etc.

The following `sources` are currently available:

.. toctree::
   :maxdepth: 1

   sources/accumulations
   sources/anemoi-dataset
   sources/cds
   sources/eccc-fstd
   sources/forcings
   sources/grib
   sources/grib-index
   sources/hindcasts
   sources/mars
   sources/netcdf
   sources/opendap
   sources/recentre
   sources/repeated-dates
   sources/xarray-based
   sources/xarray-kerchunk
   sources/xarray-zarr
   sources/zenodo
