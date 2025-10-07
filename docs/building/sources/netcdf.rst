########
 netcdf
########

In the examples below, we explain how to create an anemoi dataset from
one or more netCDF files.

.. literalinclude:: yaml/netcdf.yaml
   :language: yaml

The netCDF source uses `Xarray
<https://docs.xarray.dev/en/stable/index.html>`_ internally to access
the data, and assumes that the netcdf files follow the `CF conventions
<https://cfconventions.org/>`_. You can also read a collection of netCDF
files, using Unix’ shell `wildcards
<https://en.wikipedia.org/wiki/Glob_(programming)>`_

.. warning::

   We are aware of instances in wich the creation of an anemoi dataset
   from a netCDF source is not working as expected due to the missing
   information in the files metadata that anemoi-datasets expects.
   anemoi-datasets internal routines do their best to infer the missing
   information, but in some cases it is not possible. If you encounter
   this or similar issues, please open an issue in the anemoi-datasets
   repository.

See :ref:`create-cf-data` for more information.
