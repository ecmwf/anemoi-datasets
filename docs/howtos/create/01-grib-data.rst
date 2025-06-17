.. _create-grib-data:

#################################
 Create a dataset from GRIB data
#################################

A GRIB file is a file that contains several GRIB `messages`. Each
message is a single 2D field. `anemoi-datasets` relies earthkit-data_ to
read GRIB files, which itself relies on eccodes_.

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

.. _file-pattern:

********************************************************
 Reading GRIB messages from files that follow a pattern
********************************************************

Often, GRIB files are stored in a directory with a specific pattern. For
example, the files may be named with a date pattern, such as
``YYYYMMDD_HHMM.grib``. In this case, you can use the :ref:`grib
<grib_source>` source.

.. literalinclude:: yaml/grib-recipe2.yaml

Please note that the ``path`` keyword can also be a list.

Every pattern in the ``path`` that is enclosed in curly brackets
(``{}``) is replaced by the requested value. For example, The path
``/path/to/files/{param}_{level}.grib`` will be replaced by
``/path/to/files/z_500.grib`` if the requested parameter is ``z`` and
the level is ``500``.

There is a special syntax for the ``date`` keyword:

The construct ``{date:strftime(%Y%m%d%H)}`` is replaced by the requested
date formatted according to the Python strftime_ method. For example, if
the requested date is ``2023-01-01 00:00:00``, the pattern will be
``2023010100.grib``.

You can also use ``strftimedelta`` to specify a date that is shifted by
an offset from the requested date. For example, if you want to read a
file that is one hour before the requested date, you can use the
following pattern ``{date:strftimedelta(-1h,%Y%m%d%H)}``. This will be
replaced by ``2023010113`` if the requested date is ``2023-01-01
14:00:00``.

You can also use Unix wildcards_ to specify a pattern for the files. For
example, if the files are named with a date pattern, such as
``YYYYMMDD_HHMM.grib``, you can use the following pattern:
``/path/to/files/*{date:strftime(%Y%m%d%H)}*.grib``. The ``*`` wildcard
will match any number of characters, including none.

*********************
 Using an index file
*********************

If you have a large number of GRIB files, it may be useful to create an
index file. This file contains the list of all the GRIB messages in the
files and allows quick access to the messages without having to read the
entire file. The index file is created using the `grib-index`
:ref:`command <grib-index_command>` and uses the `grib-index`
:ref:`source <grib-index_source>`.

.. code:: bash

   anemoi-datasets grib-index --index index.db /path/to/grib-files --match '*pattern*'

The index file can then be used in the recipe file. For example, if the
index file is named ``index.db``, you can use the following recipe:

.. literalinclude:: yaml/grib-recipe3.yaml

after that, the parameters are the same as for the `grib` source.

*************************
 Selecting GRIB messages
*************************

You can select GRIB messages using the MARS language. For example, to
select all the GRIB messages with a specific parameter, you can use the
``param`` keyword. For example, to select all the GRIB messages with the
parameters ``2t``, ``10u`` and ``10v``, you can use the following:

.. literalinclude:: yaml/grib-recipe4.yaml

It is recommended to have several sources to differentiate between
single-level and multi-level fields.

.. literalinclude:: yaml/grib-recipe5.yaml

.. note::

   You can use any eccodes_ keys to select the GRIB messages. If you are
   using an index, the keys must be present in the index file, and
   should have been provided at index creation time.

.. _grib_flavour:

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

.. _earthkit-data: https://earthkit-data.readthedocs.io/en/latest/

.. _eccodes: https://github.com/ecmwf/eccodes

.. _strftime: https://python.readthedocs.io/en/latest/library/datetime.html#strftime-and-strptime-behavior

.. _wildcards: https://en.wikipedia.org/wiki/Glob_(programming)
