#############
 xarray-based
#############


The netcdf source uses Xarray_ internally to access the data, and
assumes that the netcdf files follow the `CF conventions`_. You can also
read a collection of GRIB files, using Unix’ shell wildcards:

You specify any valid xarray.open_dataset_ arguments in the source.

.. literalinclude:: yaml/xarray-based.yaml
   :language: yaml

.. _cf conventions: http://cfconventions.org/

.. _xarray: https://docs.xarray.dev/en/stable/index.html

.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
