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

   For all Xarray-based sources, the ``param`` and ``variable`` are
   considered synonymous. This is also true for the ``level`` and
   ``levelist`` keywords.

*********
 OpenDAP
*********

OpenDAP is a protocol that allows you to access remote datasets over the
internet. The OpenDAP source is identical toe the NetCDF source. The
only difference is that a URL is used instead of a file path.

.. literalinclude:: yaml/opendap1.yaml
   :language: yaml

******
 Zarr
******

.. literalinclude:: yaml/zarr1.yaml
   :language: yaml

**********
 Patching
**********

(Coming soon)

*******************
 Using a `flavour`
*******************

(Coming soon)

.. _cf conventions: https://cfconventions.org/

.. _netcdf: https://www.unidata.ucar.edu/software/netcdf/

.. _opendap: https://www.opendap.org/

.. _xarray: https://xarray.pydata.org/en/stable/

.. _zarr: https://zarr.readthedocs.io/
