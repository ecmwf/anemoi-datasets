########################
 Handling missing dates
########################

By default, the package will raise an error if there are missing dates.

Missing dates can be handled by specifying a list of dates in the
configuration file. The dates should be in the same format as the dates
in the time series. The missing dates will be filled with ``np.nan``
values.

Example with individual missing dates:

.. literalinclude:: ../yaml/missing_dates.yaml
   :language: yaml

You can also provide missing dates as ranges using ``start`` and
``end``. This is useful when long contiguous periods are missing.
The range is expanded using the dataset ``dates.frequency`` unless a
``frequency`` is provided in the range entry.

Example with missing date ranges:

.. literalinclude:: ../yaml/missing_dates_range.yaml
   :language: yaml

*Anemoi* will ignore the missing dates when computing the
:ref:`statistics <gathering_statistics>`.

You can retrieve the list indices corresponding to the missing dates by
accessing the ``missing`` attribute of the dataset object.

.. code:: python

   print(ds.missing)

If you access a missing index, the dataset will throw a
``MissingDateError``.
