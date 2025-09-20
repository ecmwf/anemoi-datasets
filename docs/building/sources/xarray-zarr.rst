#############
 xarray-zarr
#############

Here is an example recipe that builds a dataset using one of the many
regridded versions of ERA5 hosted by Google in an Analysis-Ready,
Cloud-Optimised format. See `here
<https://cloud.google.com/storage/docs/public-datasets/era5>`_ for more
information.

.. literalinclude:: yaml/xarray-zarr.yaml
   :language: yaml

Note that, unlike the ``mars`` examples, there is no need to include a
``grid`` specification. Additionally, to sub-select the vertical levels,
it is necessary to use the :ref:`join <building-join>` operation to join
separate lists containing 2D variables and 3D variables. If all vertical
levels are desired, then it is acceptable to specify a single source.

See :ref:`create-cf-data` for more information.
