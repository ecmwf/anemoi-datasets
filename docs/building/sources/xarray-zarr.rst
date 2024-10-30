#############
 xarray-zarr
#############

Here is an example recipe that builds a dataset using one of the many
regridded versions of ERA5 hosted by Google in Analysis-Ready, Cloud
Optimized format. See `here
<https://cloud.google.com/storage/docs/public-datasets/era5>`_ for more
information.

.. literalinclude:: yaml/xarray-zarr.yaml
   :language: yaml

Note that unlike the ``mars`` examples, there is no need to include a
``grid`` specification. Also, in order to subselect the vertical levels,
it is necessary to use the :ref:`join <building-join>` operation to join
separate lists containing 2D variables and 3D variables. If all vertical
levels are desired, then it is OK to specify a single source.
