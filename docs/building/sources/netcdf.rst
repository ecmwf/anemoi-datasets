########
 netcdf
########

In the examples below, we explain how to create an anemoi dataset from
one or more netCDF files.

.. literalinclude:: yaml/netcdf.yaml
   :language: yaml

The netCDF source uses `Xarray
<https://docs.xarray.dev/en/stable/index.html>`_ internally to access
the data, and assumes that the netCDF files follow the `CF conventions
<https://cfconventions.org/>`_. You can also read a collection of netCDF
files using the Unix shell `wildcards
<https://en.wikipedia.org/wiki/Glob_(programming)>`_.

.. warning::

   We are aware of instances in which the creation of an anemoi dataset
   from a netCDF source does not work as expected due to missing
   information in the files' metadata that anemoi-datasets expects. The
   anemoi-datasets' internal routines do their best to infer missing
   information, but in some cases this is not possible. If you encounter
   this or similar issues, please open an issue in the anemoi-datasets
   repository.

See :ref:`create-cf-data` for more information.
