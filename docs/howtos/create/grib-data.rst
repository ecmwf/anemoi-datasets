.. _create-grib-data:

#################################
 Create a dataset from GRIB data
#################################

***********************************
 Reading GRIB messages from a file
***********************************

To create a dataset from GRIB files, use the :ref:`grib <grib_source>`
source.

.. literalinclude:: yaml/grib-recipe1.yaml

This recipe will create a dataset with all the GRIB messages present in
the file, whose *valid date* match the requested dates. This means that
for forecast data, the date at which the data are valid is usually the
reference date of the forecast (starting date) plus the forecast step.

**********************************************************
 Reading GRIB messages from a files that follow a pattern
**********************************************************

Often, GRIB files are stored in a directory with a specific pattern. For
example, the files may be named with a date pattern, such as
``YYYYMMDD_HHMM.grib``. In this case, you can use the :ref:`grib
<grib_source>`

.. literalinclude:: yaml/grib-recipe2.yaml

Please note that the ``path`` keyword can also be a list.

*********************
 Using an index file
*********************

If you have a large number of GRIB files, it may be useful to create an
index file. This file contains the list of all the GRIB messages in the
files, and allows to quickly access the messages without having to read
the entire file. The index file is created using the ``grib-index``
:ref:`command <grib-index_command>` and use the `grib-index` source.

.. literalinclude:: yaml/grib-recipe3.yaml

*************************
 Selecting GRIB messages
*************************

You can select GRIB messages using the MARS language. For example, to
select all the GRIB messages with a specific parameter, you can use the
``param`` keyword. For example, to select all the GRIB messages with the
parameter ``2t`` (2m temperature), you can use the following

.. literalinclude:: yaml/grib-recipe4.yaml

It is recommended to have several sources to differentiate between
single-levels and multi-levels fields.

.. literalinclude:: yaml/grib-recipe5.yaml

Using a `flavour`
=================

GRIB from different organisations often have slightly different
`flavours` such as organisation-specific naming conventions, or
different ways of understanding single-levels and multi-levels fields.
