########
 netCDF
########

In the examples below we explain how to create an anemoi dataset from
one or more netCDF files.

.. literalinclude:: yaml/netcdf.yaml
   :language: yaml

The netCDF source uses Xarray_ internally to access the data, and
assumes that the netcdf files follow the `CF conventions`_. You can also
read a collection of netCDF files, using Unix’ shell wildcards_:

.. warning::

   We are aware of instances in wich the creation of an anemoi dataset
   from a netCDF source is not working as expected due to the missing
   information in the files metadata that anemoi-datasets expects.
   anemoi-datasets internal routines do their best to infer the missing
   information, but in some cases it is not possible. If you encounter
   this or similar issues, please open an issue in the anemoi-datasets
   repository.

######################
 Xarray-based Sources
######################

More in general, you can specify any valid xarray.open_dataset_
arguments as the source and anemoi-dataset will try to build a dataset
from it. Examples of valid xarray.open_dataset_ arguments are: netCDF,
zarr, opendap, etc.

.. literalinclude:: yaml/xarray-based.yaml
   :language: yaml

.. _cf conventions: http://cfconventions.org/

.. _wildcards: https://en.wikipedia.org/wiki/Glob_(programming)

.. _xarray: https://docs.xarray.dev/en/stable/index.html

.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
