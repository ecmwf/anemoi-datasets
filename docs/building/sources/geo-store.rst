.. _geo-store-source:

###########
 geo_store
###########

The ``geo_store`` source reads the geostationary day stores that
``obsbase`` writes (one zarr 3 store per day of hourly images on an
``oNNNN`` cell subset, e.g. GridSat-B1 or one geostationary slot) as
observation rows: one row per hour and cell holding a value, at the
cell's centre and the hour's nominal time. It gives a DataFrame, so it
is used as the input of :ref:`grid_observations
<grid-observations-source>`.

.. code-block:: yaml

   geo_store:
     path: /ec/project/dop/base_data/gridsat_noaa_cdr/year=%Y/month=%m/%Y-%m-%d.zarr
     satid: satid_ir
     missing_files: skip

``path`` is a strftime template of one day's store. ``satid`` names the
store variable that gives each row's satellite (the ``satid`` column).
A value is kept only where its own satellite (the variable's ``satid``
attribute, e.g. ``satid_wv``) is the row's, so a row never mixes
satellites. Only the hours inside one of the dataset dates' windows are
read. A missing day store is an error unless ``missing_files: skip``
(needed when the first window reaches into the day before the record).

The stores are read without zarr 3, in the layouts ``obsbase`` writes
(one zstd chunk per array, or one shard of single-hour zstd chunks); any
other layout is an error.
