.. _create-cf-data:

#########################################
 Create a dataset from CF-compliant data
#########################################

A CF-compliant dataset is a dataset that follows the `CF conventions`_.
These dataset are usually stored in a format that is compatible with the
CF conventions, such as NetCDF_, OpenDAP_, or Zarr_. Internally, these
datasets are accessed by `anemoi-datasets` using the Xarray_ library.

********
 NetCDF
********

(Coming soon)

.. literalinclude:: yaml/netcdf1.yaml
   :language: yaml

.. admonition:: Note

   For all Xarray-based sources, the ``param`` and ``variable`` keywords
   are considered synonymous. This is also true for the ``level`` and
   ``levelist`` keywords.

Please note that the ``path`` keyword can also be a list, and that paths
can contain wildcards and patterns. See :ref:`file-pattern` for more
information.

*********
 OpenDAP
*********

OpenDAP is a protocol that allows you to access remote datasets over the
internet. The OpenDAP source is identical toe the NetCDF source. The
only difference is that a URL is used instead of a file path.

.. literalinclude:: yaml/opendap1.yaml
   :language: yaml

Please note that the ``url`` keyword can also be a list, and that URLs
can contain patterns. See :ref:`file-pattern` for more information.

******
 Zarr
******

.. literalinclude:: yaml/zarr1.yaml
   :language: yaml

*********************************************
 Handling data that is not 100% CF-compliant
*********************************************

(Coming soon)

Patching
========

Consider the following dataset:

.. code:: console

   <xarray.Dataset> Size: 21MB
   Dimensions:   (y: 1207, x: 1442)
   Dimensions without coordinates: y, x
   Data variables:
      nav_lat   (y, x) float32 7MB ...
      nav_lon   (y, x) float32 7MB ...
      mask      (y, x) float32 7MB ...

Although the variables ``nav_lat`` and ``nav_lon`` are coordinates,
there are not marked as such. This can be fixed by using the ``patch``
keyword in the recipe file:

.. literalinclude:: yaml/xarray-patch1.yaml
   :language: yaml

The resulting dataset will look like this:

.. code:: console

   <xarray.Dataset> Size: 21MB
   Dimensions:   (y: 1207, x: 1442)
   Coordinates:
      nav_lat   (y, x) float32 7MB ...
      nav_lon   (y, x) float32 7MB ...
   Dimensions without coordinates: y, x
   Data variables:
      mask      (y, x) float32 7MB ...

.. note::

   Patching only happens in memory. The patched dataset is not saved and
   the original dataset is not modified.

Using a `flavour`
=================

(Coming soon)

.. literalinclude:: yaml/xarray-flavour1.yaml

You can see examples of the `flavour` in the following tests_.

.. _cf conventions: https://cfconventions.org/

.. _netcdf: https://www.unidata.ucar.edu/software/netcdf/

.. _opendap: https://www.opendap.org/

.. _tests: https://github.com/ecmwf/anemoi-datasets/blob/main/tests/xarray/test_zarr.py

.. _xarray: https://xarray.pydata.org/en/stable/

.. _zarr: https://zarr.readthedocs.io/
