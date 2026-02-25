#####
 odb
#####

This source reads data from an ODB_ file using ``earthkit.data`` under
the hood.

**************************
 Configuration Parameters
**************************

When configuring the ODB source in your YAML file, you can specify the
following parameters:

**select** (str, optional)
   Specifies which columns to read from the ODB file. Defaults to all
   columns (``"*"``).

**where** (str, optional)
   Filters the data based on a condition. Defaults to no filtering.

**flavour** (dict, optional)
   Defines the names of the latitude, longitude, date, and time columns
   in your data. Defaults to:

   -  ``latitude_column_name``: ``lat@hdr``
   -  ``longitude_column_name``: ``lon@hdr``
   -  ``date_column_name``: ``date@hdr``
   -  ``time_column_name``: ``time@hdr``

**pivot_columns** (list, optional)
   Lists column names whose values will become new columns after
   reshaping. These typically identify observation types such as
   ``"channel_number"`` or ``"varno"``.

**pivot_values** (list, optional)
   Lists column names whose values will be spread across different
   columns during reshaping. Common examples include
   ``"observed_value"`` and ``"quality_control_value"``.

In the example below, the date is represented in the CSV file as two
columns, named ``date`` and ``time``, so we specify them as a list.

.. literalinclude:: yaml/odb.yaml
   :language: yaml

.. _odb: https://www.ecmwf.int/en/elibrary/74516-odb-observational-database-and-its-usage-ecmwf
