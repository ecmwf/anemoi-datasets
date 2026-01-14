###############
 accumulate
###############

.. note::

   The `accumulate` source was previously named `accumulations`.
   The API has changed in the following ways:

   - The parameter `accumulation_period` has been renamed to `period`.
   - The source can be now different from `mars` (e.g., `mars`, `grib-index`)
     it must now be explicitly specified as a nested dictionary under the `source` key.
   - The (optional) available accumulation intervals can now be specified using the `availability` key.

Accumulations and flux variables, such as precipitation, are often
forecast fields, which are archived for a given base date (or reference
time) and a forecast time (or step). These fields are valid at the
forecast time and are accumulated over a given period of time, with the
relation: :math:`valid\_date = base\_date + step`.

Because the package builds datasets according to the valid date of the
fields, it must be able to reconstruct the requested accumulation period
from the available data in the source dataset. Furthermore, some fields
are accumulated since the beginning of the forecast (e.g. ECMWF
operational forecast), while others are accumulated since the last time
step (e.g. ERA5).

The `accumulate` source requires the following parameters:

- **period**: The requested accumulation period (e.g., ``6h``, ``12h``, ``24h``).
  This can be specified as a string with units (``"6h"``) (or as an integer
  representing hours (``6``)). Minutes (``"30min"``) are not supported yet.
- **source**: The data source configuration. Currently only ``mars`` and ``grib-index`` sources are supported.
- **availability** : Information about how accumulations are stored in
  the data source. This allows the package to determine which intervals to use
  for reconstructing the requested accumulation period.
  The `accumulate` source has built-in knowledge for well-known datasets
  with specific class/stream in `mars` and can infer the `availability`
  of the accumulated intervals using the value `availability: auto`.
  If the package cannot automatically determine the accumulation intervals,
  or if you need a finer control over the reconstruction process,
  you can provide this information manually using the `availability`
  parameter (see below).

.. note::

   If the data provided byt the source does not match the definition provided
   in the `availability` parameter, the package will attempt to check the
   metadata of the source dataset and fail if the accumulation periods cannot
   be reconstructed.
   If the metadata is incomplete or inconsistent, the package may not be able to
   reconstruct the requested accumulation period **or may not be able to check
   the validity of the accumulation and may produce incorrect results**.


Specifying the ``availability`` of accumulation intervals
=========================================================

There are multiple ways to specify the ``availability`` parameter:

Data accumulation methods differ between datasets. Two common methods are to
accumulate data either from the start of the forecast or from the previous time step.

-  For ECMWF operational forecasts, the data is accumulated from the
   beginning of the forecast. For example, if the accumulation period is
   6h and the valid date is 2020-01-01 00:00, the source will use the
   forecast [1]_ of 2019-12-31 18:00 at step 6h.

-  For ERA5, the data is accumulated since the last time step (hourly
   accumulations), and forecasts are only available at 06Z and 18Z. For a
   6h accumulation with valid date 2020-01-01 13:00, the source will sum
   the fields from the forecast of 2020-01-01 06:00 at steps 1-2h, 2-3h,
   3-4h, 4-5h, 5-6h, and 6-7h.

Option 1: Automatic detection
-----------------------------

The simplest approach is to use ``availability: auto``. The package will
infer the availability from the ``mars`` source parameters (class, stream, origin).

Automatic detection is not supported for ``grib-index`` source.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ECMWF operational (accumulated from start)
     - ERA5 (accumulated from previous step)
   * - .. literalinclude:: yaml/accumulations-from-start-mars-ecmwf-operational-forecast-1.yaml
          :language: yaml
     - .. literalinclude:: yaml/accumulations-from-previous-step-mars-era5-1.yaml
          :language: yaml

Option 2: Type-based configuration
----------------------------------

For more explicit control, use the ``type`` parameter with ``accumulated-from-start``
or ``accumulated-from-previous-step``, along with ``basetime``, ``frequency``, and ``last_step``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ECMWF operational (accumulated from start)
     - ERA5 (accumulated from previous step)
   * - .. literalinclude:: yaml/accumulations-from-start-mars-ecmwf-operational-forecast-2.yaml
          :language: yaml
     - .. literalinclude:: yaml/accumulations-from-previous-step-mars-era5-2.yaml
          :language: yaml

Option 3: Explicit list of intervals
------------------------------------

For full control, provide an explicit list of ``(basetime, steps)`` pairs.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ECMWF operational (accumulated from start)
     - ERA5 (accumulated from previous step)
   * - .. literalinclude:: yaml/accumulations-from-start-mars-ecmwf-operational-forecast-3.yaml
          :language: yaml
     - .. literalinclude:: yaml/accumulations-from-previous-step-mars-era5-3.yaml
          :language: yaml

Option 4: accumulation available over a fixed period
----------------------------------------------------

If the source provides data accumulated over a fixed period, such as
``availability: "1h"`` for hourly accumulated data, ``"3h"`` for
3-hourly accumulated data, etc.

This approach should be used when all accumulation intervals for the fixed period are available
for all base times.

Additionally, the period provided in ``availability`` must be compatible with the requested accumulation period,
i.e., it must be a divisor of the requested period in ``period``.

.. literalinclude:: yaml/accumulations-grib-index.yaml
   :language: yaml

.. [1]

   For ECMWF forecasts, the forecasts at 00Z and 12Z are from the stream
   `oper` while the forecasts at 06Z and 18Z are from the stream `scda`.
