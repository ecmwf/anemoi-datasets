.. _parquet-source:

#########
 parquet
#########

This source reads observation rows from per-day parquet files into a
DataFrame. Only the files a time range touches are opened, and the time
filter is pushed down to ``pyarrow``, so files written with one row group
per hour skip the hours outside the range.

.. literalinclude:: yaml/parquet.yaml
   :language: yaml

**path**
   A strftime template of one day's file.

**flavour** (dict, optional)
   The file columns holding ``date``, ``latitude`` and ``longitude``.
   They are renamed to those names.

**columns** (list, optional)
   Columns to read, by their output names. A calling source such as
   :ref:`grid_observations <grid-observations-source>` adds the columns it
   needs. With neither, every column is read.

**window** (str, optional)
   Needed only when the source is called with dates rather than a time
   range: the rows of ``(first date + start, last date + end]``.

**missing_files** (str)
   ``error`` (the default) or ``skip``. With ``skip``, a day without a
   file has no rows.

Needs ``pyarrow`` (the ``observations`` extra).
