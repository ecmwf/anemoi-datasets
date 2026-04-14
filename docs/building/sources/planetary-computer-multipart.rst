#############################
 planetary-computer-multipart
#############################

This source is for building datasets from STAC collections on the `open
Microsoft Planetary Computer <https://planetarycomputer.microsoft.com/>`_.

It's intended for collections containing multiple items and, potentially,
multiple assets per item. Conversely, the ``planetary-computer`` source is for
those with a collection-level dataset asset under the ``zarr-abfs`` key
corresponding to a single Zarr store containing all data.

Below is an example recipe that builds a dataset using the `ERA5 - PDS
<https://planetarycomputer.microsoft.com/dataset/era5-pds>`_ collection.

.. literalinclude:: yaml/planetary-computer-multipart.yaml
   :language: yaml

The ``query`` config section enables specification of filters for the STAC
items to include in the dataset. To identify a collection's queryable fields,
visit its queryables endpoint (e.g., `ERA5 - PDS queryables
<https://planetarycomputer.microsoft.com/api/stac/v1/collections/era5-pds/queryables>`_)
or use the Python equivalent
``pystac_client.CollectionClient(...).get_queryables``.
While not strictly necessary, it is recommended to include a datetime filter
under ``query.datetime`` to reduce query time and the number of results to
filter. See `pystac_client.Client.search
<https://pystac-client.readthedocs.io/en/v0.7.6/_modules/pystac_client/client.html#Client.search>`_
for accepted formats.
