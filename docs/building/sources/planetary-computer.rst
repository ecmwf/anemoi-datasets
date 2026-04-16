#############################
 planetary-computer
#############################

This source is for building datasets from STAC collections on the `open
Microsoft Planetary Computer <https://planetarycomputer.microsoft.com/>`_.

It's intended for two types of collections:

1. Containing a collection-level dataset asset under the ``zarr-abfs`` key
corresponding to a single Zarr store containing all data.

2. Containing multiple items and, potentially, multiple assets per item.

Below is an example recipe that builds a dataset using the `Near-surface level
collection Met Office global deterministic 10km forecast
<https://planetarycomputer.microsoft.com/dataset/met-office-global-deterministic-near-surface>`_
collection.

.. literalinclude:: yaml/planetary-computer.yaml
   :language: yaml

**The following is applicable to collections with multiple items and assets
only.**

The ``search_params`` config section enables specification of mappings and
filters for the STAC items and assets to include in the dataset. Supported
parameters include:

- ``datetime``: passed to the STAC API to filter items by their datetime
  field(s).
- ``variable_key_map``: a mapping of data variable names to STAC asset keys
  **for collections where they differ**.
- ``filter``: a CQL2 filter (dict for cql2-json, string for cql2-text) passed
  directly to the STAC API to filter items server-side.

.. tip::
  While not required, it is recommended to include a datetime filter under
  ``search_params.datetime`` to reduce query time and the number of results to
  filter. See `pystac_client.Client.search
  <https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.search>`_
  for accepted formats.

.. tip::
  To identify a collection's queryable fields, visit its queryables endpoint
  (e.g., `ERA5 - PDS queryables
  <https://planetarycomputer.microsoft.com/api/stac/v1/collections/era5-pds/queryables>`_
  ) or use the Python equivalent
  ``pystac_client.CollectionClient(...).get_queryables``.

See other example recipes in the tests_.

.. _tests: https://github.com/ecmwf/anemoi-datasets/blob/main/tests/create/test_sources.py
