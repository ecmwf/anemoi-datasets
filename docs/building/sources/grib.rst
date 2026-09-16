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

.. _grib-path-keywords:

Path keywords
=============

``date`` is the *validity* time, and is available in every recipe. Some
archives name their files after something else — a forecast archive by
its model run and lead time, an accumulation archive by the bounds of
the window — so the following keywords are also available, depending on
what the recipe asks the source for:

.. list-table::
   :header-rows: 1

   -  -  Keyword
      -  Meaning
      -  Available in
   -  -  ``date``
      -  Validity time. For an accumulation, the end of the window.
      -  every recipe
   -  -  ``base_date``
      -  Model-run (initialisation) time.
      -  ``layout: trajectories``, and ``accumulate:`` over a forecast
         archive
   -  -  ``step``
      -  Lead time in whole hours, ``date − base_date``.
      -  as ``base_date``
   -  -  ``step_minutes``
      -  The same lead time in whole minutes. Use it when the archive
         names its files by a sub-hourly lead time.
      -  as ``base_date``
   -  -  ``step_seconds``
      -  The same lead time in whole seconds.
      -  as ``base_date``
   -  -  ``start_date``
      -  Start of the accumulation window.
      -  ``accumulate:``
   -  -  ``end_date``
      -  End of the accumulation window (same as ``date``).
      -  ``accumulate:``
   -  -  ``middle_date``
      -  Midpoint of the accumulation window.
      -  ``accumulate:``

``base_date``, ``start_date``, ``end_date`` and ``middle_date`` are dates,
formatted with strftime_ exactly like ``date``.

``step`` is an integer, so it takes a printf integer format — use it to
match the archive's zero padding:

.. list-table::
   :header-rows: 1

   -  -  Format
      -  step 0
      -  step 1
      -  step 12
      -  step 123
   -  -  ``{step:int(%d)}``
      -  ``0``
      -  ``1``
      -  ``12``
      -  ``123``
   -  -  ``{step:int(%02d)}``
      -  ``00``
      -  ``01``
      -  ``12``
      -  ``123``
   -  -  ``{step:int(%03d)}``
      -  ``000``
      -  ``001``
      -  ``012``
      -  ``123``

The width is a minimum, not a truncation: a step that outgrows it widens
rather than being cut, so an archive whose steps pass ``99`` keeps
working.

For example, a forecast archive with one directory per run and one file
per lead time is read as:

.. literalinclude:: yaml/grib5.yaml
   :language: yaml

A keyword is only substituted when the recipe gives the source the
information it needs. Asking for ``{step}`` where no model-run time is
known — a plain ``dates:`` recipe, or an accumulation over a
validity-time-indexed archive — raises ``Missing parameter 'step'``.
A lead time that is not a whole number of hours is refused rather than
truncated (``expected an integer``), since truncating it would silently
read a neighbouring file. Use ``{step_minutes}`` (or ``{step_seconds}``)
for an archive whose files are named by a sub-hourly lead time: those two
keywords always render, whatever the lead time.

Each archive field is read once, however many output rows it feeds. An
accumulation that both adds and subtracts the same field, and
overlapping trajectory rows that share a window, do not re-read it.

.. _grib-or-grib-index:

grib or grib-index?
===================

`grib` and :ref:`grib-index <grib-index_source>` both read local GRIB.
They differ in how a request finds its field:

-  `grib` builds a file path from the keywords above, then opens that
   file and selects within it. It needs no preparation, but the recipe
   has to encode the archive's naming scheme, and each request parses a
   whole file.
-  `grib-index` looks the message up in an index built once with the
   :ref:`grib-index command <grib-index_command>`, then reads just that
   message by byte offset. The recipe carries no paths at all, so
   irregular naming, several runs per file, or a reorganised archive cost
   nothing — and large files are not re-parsed per request.

For a forecast archive feeding a :ref:`trajectory
<layouts-trajectories>`, `grib-index` is usually the better of the two:
the index records the model run, so a row is addressed without the
recipe having to describe file names or guess how the archive encodes
its step.

Both also work under ``accumulate:``, where each archived field is
addressed by whichever terms the archive itself provides — a field
belonging to a model run by that run, a field from a
validity-time-indexed archive by its accumulation length. The covering
decides which applies; the recipe does not have to say.

.. note::

   If an archive holds several accumulation windows for the same
   parameter and validity time within one run, add a selector that tells
   them apart (e.g. ``timespan: fs``). Otherwise both are returned and
   the accumulator reports a field it could not use.

See :ref:`create-grib-data` for more information.

.. note::

   You can combine all the above options when selecting GRIB messages
   from a file.

.. _strftime: https://python.readthedocs.io/en/latest/library/datetime.html#strftime-and-strptime-behavior

.. _wildcards: https://en.wikipedia.org/wiki/Glob_(programming)
