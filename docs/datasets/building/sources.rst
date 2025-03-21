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

   sources/accumulations
   sources/eccc_fstd
   sources/forcings
   sources/grib
   sources/hindcasts
   sources/mars
   sources/cds
   sources/netcdf
   sources/opendap
   sources/recentre
   sources/repeated_dates
   sources/xarray-kerchunk
   sources/xarray-zarr
   sources/xarray-based
   sources/zenodo
