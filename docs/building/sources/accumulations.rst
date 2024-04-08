###############
 accumulations
###############

.. note::

   The `accumulations` source is currently using the `mars` source
   internally. This will be changed in the future.

Accumulations and fluxes variables, such as precipitations, are often
forecast fields, which are archived for a given a base date (or
reference time) and a forecast time (or step). These fields are valid at
the forecast time, and are accumulated over a given period of time, with
the relation: :math:`valid\_date = base\_date + step`.

Because the package build datasets according to the valid date of the
fields, it must be able to compute the base date and the forecast time
from the valid date. Furthermore, some fields are accumulated since the
beginning of the forecast (e.g. ECMWF operational forecast), while
others are accumulated since the last time step (e.g. ERA5).

The `accumulations` has some of that knowledge built-in. If the source
dataset is unknown, the package assumes that the fields to use are
accumulated since the beginning of the forecast, over a 6h period.

The user can specify the desired accumulation period with the
``accumulation_period`` parameter. If its value is a single interger,
the source will attempt to accumulate the variables over that period.
This does not always mean that the data used is accumulated from the
beginning of the forecast, but the most recent data available will be
used:

-  For ECMWF operational forecasts, the data is accumulated from the
   beginning of the forecast. So if the accumulation period is 6h, for
   the date 2020-01-01 00:00, the source will use the forecast[1]_ of
   2019-12-31 18:00 and the step 6h.

If the value is a pair of integers, the source will attempt to
accumulate the variables over each period specified by the pair of
integers.

.. literalinclude:: accumulations1.yaml
   :language: yaml

or:

.. literalinclude:: accumulations2.yaml
   :language: yaml

.. [1]

   For ECMWF forecasts, the forecasts at 00Z and 12Z are from the stream
   `oper` while the forecasts at 06Z and 18Z are from the stream `scda`.
