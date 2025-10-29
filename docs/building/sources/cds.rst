.. _cds_source:

#####
 cds
#####

For users outside of the ECMWF organisation, it is possible to access
ERA5 data through the Copernicus Climate Data Store ``cdsapi`` instead.

The steps to set up a CDS API account are detailed `here
<https://cds.climate.copernicus.eu/how-to-api>`_.

The only difference with the previous MARS recipes is the addition of
the ``use_cdsapi_dataset`` key:

.. literalinclude:: yaml/mars-cds.yaml
   :language: yaml

This process can take some time because of the high demand on the CDS
server.
