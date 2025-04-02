.. _grib_source:

######
 grib
######

To read all the GRIB from a file, use the following:

.. literalinclude:: yaml/grib1.yaml
   :language: yaml

You can also read specific GRIB messages by specifying them using the
MARS language (excluding the keywords `date`, `time`, and `step`, as
well as any post-processing options, such as `grid` or `area`):

.. literalinclude:: yaml/grib2.yaml
   :language: yaml

You can also read a collection of GRIB files, using Unix shell
wildcards_:

.. literalinclude:: yaml/grib3.yaml
   :language: yaml

You can also use the requested `date` to build the filenames. For
example, if the GRIB files containing the requested data are named
according to the following pattern: ``/path/to/YYYY/MM/YYYYMMDDHH.grib``
with `YYYY` being the year, `MM` the month, `DD` the day, and `HH` the
hour, you can use the following configuration:

.. literalinclude:: yaml/grib4.yaml
   :language: yaml

The patterns in between the curly brackets are replaced by the values of
the `date` and formatted according to the Python strftime_ method.

See :ref:`create-grib-data` for more information.

.. note::

   You can combine all the above options when selecting GRIB messages
   from a file.

.. _strftime: https://python.readthedocs.io/en/latest/library/datetime.html#strftime-and-strptime-behavior

.. _wildcards: https://en.wikipedia.org/wiki/Glob_(programming)
