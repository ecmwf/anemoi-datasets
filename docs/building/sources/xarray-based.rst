######################
 xarray-based Sources
######################

More generally, you can specify any valid xarray.open_dataset_ arguments
as the source, and anemoi-dataset will try to build a dataset from it.
Examples of valid xarray.open_dataset_ arguments include: netCDF, Zarr,
OpenDAP, etc.

.. literalinclude:: yaml/xarray-based.yaml
   :language: yaml

See :ref:`create-cf-data` for more information.

.. _cf conventions: http://cfconventions.org/

.. _wildcards: https://en.wikipedia.org/wiki/Glob_(programming)

.. _xarray: https://docs.xarray.dev/en/stable/index.html

.. _xarray.open_dataset: https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html
