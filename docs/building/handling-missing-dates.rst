########################
 Handling missing dates
########################

By default, the package will raise an error if there are missing dates.

Missing dates can be handled by specifying a list of dates in the
configuration file. The dates should be in the same format as the dates
in the time series. The missing dates will be filled with ``np.nan``
values.

.. literalinclude:: ../yaml/missing_dates.yaml
   :language: yaml

*Anemoi* will ignore the missing dates when computing the
:ref:`statistics <gathering_statistics>`.

You can retrieve the list indices corresponding to the missing dates by
accessing the ``missing`` attribute of the dataset object.

.. code:: python

   print(ds.missing)

If you access a missing index, the dataset will throw a
``MissingDateError``.

***********************************
 Trajectories: missing base dates
***********************************

The same ``missing:`` key applies to ``base_dates:`` for the trajectories
layout.  The list contains the base (initialisation) dates whose
``(basetime, step)`` pairs should be skipped at build time.  The
corresponding axis-0 slot is still allocated on disk and filled with
``np.nan``; reading it raises ``MissingDateError`` (see
:ref:`selecting-missing`).

.. literalinclude:: ../yaml/missing_base_dates.yaml
   :language: yaml

***********************************
 Calendar filters on dates
***********************************

In addition to ``start`` / ``end`` / ``frequency`` you can restrict the
generated dates to specific weekdays or days of the month.  The same keys
work for ``dates:`` (gridded / tabular) and ``base_dates:`` (trajectories):

- ``weekday:`` accepts a single weekday name or a list — e.g.
  ``[tuesday, friday]``.
- ``date:`` accepts a wildcard pattern in the shape ``YYYY-MM-DD``, where
  any position can be replaced by ``?`` to mean "any value".  Examples:
  ``????-??-01`` (first of every month), ``????-06-??`` (every day of
  June), or a list combining several patterns.
- ``month:`` accepts a single month or a list (numeric or name).

These filters compose: ``weekday`` + ``date`` + ``month`` is the
intersection of the three.  Filtered dates are entirely absent from the
dataset (no slot reserved); use ``missing:`` for dates that should keep
their slot but be flagged as missing.
