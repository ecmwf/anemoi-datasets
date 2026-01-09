###############
 accumulate
###############

.. note::

   The `accumulate` source was previously named `accumulations`.
   The parameter `accumulation_period` has been renamed to `period`.
   The source (e.g., `mars`, `grib-index`) must now be explicitly
   specified as a nested dictionary under the `source` key.


Accumulations and flux variables, such as precipitation, are often
forecast fields, which are archived for a given base date (or reference
time) and a forecast time (or step). These fields are valid at the
forecast time and are accumulated over a given period of time, with the
relation: :math:`valid\_date = base\_date + step`.

Because the package builds datasets according to the valid date of the
fields, it must be able to compute the base date and the forecast time
from the valid date. Furthermore, some fields are accumulated since the
beginning of the forecast (e.g. ECMWF operational forecast), while
others are accumulated since the last time step (e.g. ERA5).

The `accumulate` source requires the following parameters:

- **period**: The requested accumulation period (e.g., ``6h``, ``12h``, ``24h``).
  This can be specified as a string with units (``"6h"``) or as an integer
  representing hours (``6``).
- **source**: The data source configuration, such as ``mars`` or ``grib-index``.
  Currently only ``mars`` and ``grib-index`` sources are supported.
- **available** (optional): Information about how accumulations are stored in
  the source dataset. This helps the package determine which intervals to use
  for reconstructing the requested accumulation period.

The `accumulate` source has built-in knowledge for well-known datasets:

- For ECMWF operational forecasts (``mars`` source with specific class/stream
  combinations), the package automatically determines the appropriate intervals.
- For other sources or when automatic detection is not possible, use the
  ``available`` parameter to specify how accumulations are stored in the source.

.. note::

   Requesting accumulation periods strictly longer than 48h may lead to
   unexpected results, depending on the source dataset structure.

Example with MARS source
=========================

.. literalinclude:: yaml/accumulations-mars-1.yaml
   :language: yaml

The data used is accumulated from the beginning of the forecast, but the
most recent data available will be used:

-  For ECMWF operational forecasts, the data is accumulated from the
   beginning of the forecast. For example, if the accumulation period is
   6h and the valid date is 2020-01-01 00:00, the source will use the
   forecast [1]_ of 2019-12-31 18:00 at step 6h.

-  For ERA5, the data is accumulated since the last time step (hourly
   accumulations), and forecasts are only available at 06Z and 18Z. For a
   6h accumulation with valid date 2020-01-01 13:00, the source will sum
   the fields from the forecast of 2020-01-01 06:00 at steps 1-2h, 2-3h,
   3-4h, 4-5h, 5-6h, and 6-7h.

Using the ``available`` parameter
==================================

When the package cannot automatically determine how to reconstruct the
accumulation from the source dataset, you can provide this information
with the ``available`` parameter.

Simple case: Fixed accumulation period
---------------------------------------

If the source provides data accumulated over a fixed period, specify that
period as a string:

- ``"1h"`` for hourly accumulated data
- ``"3h"`` for 3-hourly accumulated data
- ``"6h"``, ``"12h"``, ``"24h"`` etc. for other fixed periods

Example with grib-index source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: yaml/accumulations-grib-index.yaml
   :language: yaml

.. note::

   Even with the ``available`` parameter, the package may not be able to
   reconstruct the requested accumulation period if the source dataset has
   inconsistent or incomplete metadata.

Advanced case: Specifying available intervals per base time
------------------------------------------------------------

For more complex scenarios where different base times have different
available accumulation intervals, you can provide a detailed description
using the ``available`` parameter.

Example with base-time-specific intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: yaml/accumulations-mars-2.yaml
   :language: yaml

In this example:

- For forecasts with base time 00Z, the available intervals are [0-6] and [6-12]
- For forecasts with base time 12Z, the available intervals are [12-18] and [18-24]

The package will use these intervals to reconstruct the requested 6-hour
accumulation for each valid date.

Legacy API (deprecated)
=======================

The old ``accumulations`` source (note the 's') used a different API:

.. code-block:: yaml

   input:
     accumulations:
       accumulation_period: 6
       class: ea
       param: [tp, cp, sf]
       levtype: sfc

This API is deprecated. Please use the new ``accumulate`` source with the
``period`` parameter and nested ``source`` dictionary as shown in the
examples above.

.. [1]

   For ECMWF forecasts, the forecasts at 00Z and 12Z are from the stream
   `oper` while the forecasts at 06Z and 18Z are from the stream `scda`.
