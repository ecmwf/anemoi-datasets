.. _odb-source:

#####
 odb
#####

This source reads data from an ODB_ file using ``earthkit.data`` under
the hood.

When configuring the ODB source in your YAML file, you can specify the
following parameters, in addition to the ``path`` parameter:

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

   ODB times are whole seconds. ``milliseconds_column_name`` names a
   column of milliseconds to add to them; the column is dropped from the
   output.

**pivot_columns** (list, optional)
   Lists column names whose values will become new columns after
   reshaping. These typically identify observation types such as
   ``"channel_number"`` or ``"varno"``.

**pivot_values** (list, optional)
   Lists column names whose values will be spread across different
   columns during reshaping. Common examples include
   ``"observed_value"`` and ``"quality_control_value"``.

**window** (str)
   When the source is called with dates, the rows read for each date,
   e.g. ``"[-3h,3h)"``.

.. literalinclude:: yaml/odb.yaml
   :language: yaml

*********************************
 Reading a time span (per cycle)
*********************************

A source such as :ref:`grid_observations <grid-observations-source>`
calls ``odb`` with a ``TimeSpan`` instead of dates, and then ``window`` is
not needed. The path pattern is expanded with the span's dates, e.g. one
file per cycle as written by a real-time extraction. The SQL selects the
whole seconds around the span, and the exact span is then applied to the
rows, including sub-second times when ``milliseconds_column_name`` is
set:

.. code:: yaml

   input:
     grid_observations:
       window: "(-350m,+10m]"
       # ...
       input:
         odb:
           path: /work/{date:strftime(%Y%m%d%H)}/extracted/iasi.odb
           select: "*"
           flavour:
             date_column_name: date
             time_column_name: time
             latitude_column_name: lat
             longitude_column_name: lon
             milliseconds_column_name: time_ms

.. _odb: https://www.ecmwf.int/en/elibrary/74516-odb-observational-database-and-its-usage-ecmwf
