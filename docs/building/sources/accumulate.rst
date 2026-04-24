.. _sources-accumulate:

###############
 accumulate
###############

.. note::

   The ``accumulate`` source was previously named ``accumulations``.
   The API has changed in the following ways:

   - The parameter ``accumulation_period`` has been renamed to ``period``.
   - The source can be now different from ``mars`` (e.g., ``mars``, ``grib-index``)
     it must now be explicitly specified as a nested dictionary under the ``source`` key.
   - The available accumulation intervals used to be specified using an
     ``availability`` key. That key is now **deprecated** and replaced by
     the discriminator form ``covering: { auto: <value> }`` (see below).
     The ``anemoi datasets recipe migrate`` command rewrites old recipes
     automatically.

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

The ``accumulate`` source requires the following parameters:

- **period**: The requested accumulation period (e.g., ``6h``, ``12h``, ``24h``, ``1d``).
  This can be specified as a string with units ``"6h"``.
  Periods shorter than one hour such as ``"30min"`` are not supported yet.
- **source**: The data source configuration. Currently only ``mars`` and ``grib-index`` sources are supported.
- **covering**: Information about how accumulations are stored in the
  data source, in discriminator form. The current form is
  ``covering: { auto: <value> }`` where ``<value>`` accepts the four
  shapes described under `Specifying covering of accumulation intervals`_
  below. The legacy key name ``availability:`` is accepted for one
  release with a ``DeprecationWarning``.
- **accumulation** (trajectory recipes only): one of ``from-zero`` or
  ``from-previous-step``; see `Forecast accumulations (trajectory recipes)`_.
- **patch** (optional): Patches to apply to fields returned by the source to fix metadata issues.
  Default patching is to set ``startStep`` to ``0`` when ``startStep==endStep``.

  .. warning::

     If the data provided by the source does not match the definition provided
     in the ``availability`` parameter, the package will attempt to check the
     metadata of the source dataset and fail if the accumulation periods cannot
     be reconstructed.
     Defining the period to use to reconstruct the request accumulation period and
     checking the validity of the accumulation and relies on the metadata provided by the data source.
     **If the metadata is incomplete or inconsistent, the package may produce incorrect results.**


Specifying covering of accumulation intervals
=============================================

.. note::

   The recipe key has been renamed from ``availability`` to ``covering``
   and now takes a discriminator form:

   .. code:: yaml

      accumulate:
        period: 6h
        covering: {auto: <value>}
        source: {mars: {...}}

   All four value shapes documented below are passed as the ``auto:``
   discriminator value. Other discriminators (``cycle:``) are reserved
   for future use; the ``forecast:`` discriminator is explicitly
   rejected — forecast accumulations are selected implicitly by using
   ``accumulate:`` inside a trajectory recipe (see
   `Forecast accumulations (trajectory recipes)`_). The legacy
   ``availability:`` key is accepted for one release with a
   ``DeprecationWarning``.

Historical note: the section and examples below still use the word
"availability" for the value shapes, which is correct — the covering
layer is the strategy that *uses* the availability description to pick
intervals.

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

There are multiple ways to specify the ``availability`` parameter:

- `Option 1: Type-based availability`_
- `Option 2: Availability over fixed periods`_
- `Option 3: Automatic detection for well-known datasets`_
- `Option 4: Finer control using explicit list of interval`_


Option 1: Type-based availability
---------------------------------

For more explicit control, use the **type** parameter with ``accumulated-from-start``
or ``accumulated-from-previous-step``, along with **basetime**, **frequency**, and **last_step**.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ECMWF operational (accumulated from start)
     - ERA5 (accumulated from previous step)
   * - .. literalinclude:: yaml/accumulations-from-start-mars-ecmwf-operational-forecast-2.yaml
          :language: yaml
     - .. literalinclude:: yaml/accumulations-from-previous-step-mars-era5-2.yaml
          :language: yaml

Option 2: Availability over fixed periods
-----------------------------------------

If the source provides data accumulated over a fixed period, such as
``availability: "1h"`` for hourly accumulated data, ``"3h"`` for
3-hourly accumulated data, etc.

This approach should be used when all accumulation intervals for the fixed period are available
for all base times.

Additionally, the period provided in ``availability`` must be compatible with the requested accumulation period,
i.e., it must be a divisor of the requested period in ``period``.

.. literalinclude:: yaml/accumulations-grib-index.yaml
   :language: yaml

Option 3: Automatic detection for well-known datasets
-----------------------------------------------------

The simplest approach is to use ``availability: auto``. The package will try to
infer the availability from the ``mars`` source parameters (class, stream, origin).
Supported combinations are:

- ERA5 reanalysis (class ``ea``, stream ``oper``)
- ERA5 ensemble data assimilation (class ``ea``, stream ``enda``)
- ECMWF operational forecasts (class ``od``, stream ``oper``)
- ECMWF operational ensemble data assimilation (class ``od``, stream ``elda``)
- Regional reanalysis (class ``rr``, stream ``oper``, origin ``se-al-ec``).
- ERA5-Land (class ``l5``, stream ``oper``)

Automatic detection is not supported for the ``grib-index`` source.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ECMWF operational (accumulated from start)
     - ERA5 (accumulated from previous step)
   * - .. literalinclude:: yaml/accumulations-from-start-mars-ecmwf-operational-forecast-1.yaml
          :language: yaml
     - .. literalinclude:: yaml/accumulations-from-previous-step-mars-era5-1.yaml
          :language: yaml


Option 4: Finer control using explicit list of interval
-------------------------------------------------------

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

These two examples are equivalent to those shown in Option 1 above.

Controlling the fields regrouped within accumulation
====================================================

It is possible to control the fields accumulated together through their metadata.
The ``group_by`` keyword allows to ignore some metadata when deciding to group field to accumulate them together.
Ignored keys mean that fields with different values will be accumulated together.
Note that ``date,time,step`` should be ignored by default.

.. literalinclude:: yaml/accumulations-mars-groupby.yaml
   :language: yaml


Forecast accumulations (trajectory recipes)
===========================================

Inside a :ref:`trajectory recipe <layouts-trajectories>`, ``accumulate:``
produces per-step accumulation fields anchored on the caller-imposed
basetime. The ``covering:`` key used in archive accumulations is **not**
used; the covering is determined by the basetime and a new flag:

- **accumulation**: required; one of ``from-zero`` or
  ``from-previous-step``.

  -  ``from-zero`` — the archive stores accumulations from the basetime
     (``a(0, step)``). A window ``[bt + sA, bt + sE]`` is built as
     ``+a(0, sE) − a(0, sA)``.
  -  ``from-previous-step`` — the archive stores per-step increments
     (``a(step − period, step)``). The window is a single interval.

If ``accumulation:`` is omitted in a trajectory recipe, the source
raises an error at build time. Conversely, declaring
``covering: { forecast: ... }`` explicitly is **not supported** — the
forecast branch is selected by the pipeline's argument type, not by the
recipe.

Example (extracted from ``tests/create/trajectories_accumulation.yaml``):

.. code:: yaml

   base_dates: {start: 2021-01-01, end: 2021-01-03, frequency: 12h}
   steps:      {start: 6, end: 30, frequency: 3h}

   input:
     join:
       - pipe:
           - accumulate:
               period: 1h
               accumulation: from-zero
               source:
                 mars:
                   type: fc
                   expver: "0001"
                   class: od
                   grid: 20./20.
                   param: [tp]
                   levtype: sfc
                   stream: oper
           - rename:
               param: {tp: tp_accum_1h}

   output:
     layout: trajectories


.. [1]

   For ECMWF forecasts, the forecasts at 00Z and 12Z are from the stream
   ``oper`` while the forecasts at 06Z and 18Z are from the stream ``scda``.
