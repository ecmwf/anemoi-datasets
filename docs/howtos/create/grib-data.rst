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
the file, whose *valid date* matches the requested dates. This means
that for forecast data, the date at which the data is valid is usually
the reference date of the forecast (starting date) plus the forecast
step.

Please note that the ``path`` keyword can also be a list.

********************************************************
 Reading GRIB messages from files that follow a pattern
********************************************************

Often, GRIB files are stored in a directory with a specific pattern. For
example, the files may be named with a date pattern, such as
``YYYYMMDD_HHMM.grib``. In this case, you can use the :ref:`grib
<grib_source>` source.

.. literalinclude:: yaml/grib-recipe2.yaml

Please note that the ``path`` keyword can also be a list.

You can also use ``strftimedelta`` to specify a date that is not the
current requested date.

*********************
 Using an index file
*********************

If you have a large number of GRIB files, it may be useful to create an
index file. This file contains the list of all the GRIB messages in the
files and allows quick access to the messages without having to read the
entire file. The index file is created using the ``grib-index``
:ref:`command <grib-index_command>` and uses the `grib-index` source.

..
   code: :: bash

   anemoi-datasets grib-index --index index.db /path/to/grib-file --match '*pattern*'

.. literalinclude:: yaml/grib-recipe3.yaml

*************************
 Selecting GRIB messages
*************************

You can select GRIB messages using the MARS language. For example, to
select all the GRIB messages with a specific parameter, you can use the
``param`` keyword. For example, to select all the GRIB messages with the
parameter ``2t`` (2m temperature), you can use the following:

.. literalinclude:: yaml/grib-recipe4.yaml

It is recommended to have several sources to differentiate between
single-level and multi-level fields.

.. literalinclude:: yaml/grib-recipe5.yaml

*******************
 Using a `flavour`
*******************

GRIB from different organisations often have slightly different
`flavours`, such as organisation-specific naming conventions or
different ways of understanding single-level and multi-level fields.

A `flavour` is a list of pairs of dictionaries, where the first
dictionary is a matching rule (condition) and the second one is an
action (conclusion).

When looking up fields' metadata, like the parameter name (``param``) or
the level (``level``), the first rule that matches the existing field
metadata is applied. The values listed in its second dictionary are then
used to override the actual metadata values of the field.

For example, the first rule in the example below will clear the
``levelist`` metadata fields that have a ``levtype`` of ``sfc``. This is
useful because the default naming of the variables in the resulting
dataset is the concatenation of the ``param`` and ``levelist`` fields.
If the ``level`` field is empty, the resulting variable name will be
just the ``param``. This is useful to avoid having a variable name like
``2t_2`` or ``10u_10``.

.. literalinclude:: yaml/grib-flavour1.yaml

The second and third rules will allow a user to define a ``param`` name
if the field is not recognised by eccodes_.

In a recipe file, the `flavour` can be either defined by giving a path
to a YAML or JSON file:

.. literalinclude:: yaml/grib-flavour2.yaml

or can be given inline in the recipe file.

.. literalinclude:: yaml/grib-flavour3.yaml

You can make use of YAML anchors to avoid repeating the same rules in
multiple places:

.. literalinclude:: yaml/grib-flavour4.yaml

.. _eccodes: https://github.com/ecmwf/eccodes
