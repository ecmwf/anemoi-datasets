######################
 xarray-based-sources
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
