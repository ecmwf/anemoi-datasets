###############
 accumulations
###############

.. note::

   The `accumulations` source is currently using the `mars` source
   internally. This will be changed in the future.

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

The `accumulations` has some of that knowledge built-in. If the source
dataset is unknown, the package assumes that the fields to use are
accumulated since the beginning of the forecast, over a 6h period.

The user can specify the desired accumulation period with the
``accumulation_period`` parameter. If its value is a single integer, the
source will attempt to accumulate the variables over that period. This
does not always mean that the data used is accumulated from the
beginning of the forecast, but the most recent data available will be
used:

-  For ECMWF operational forecasts, the data is accumulated from the
   beginning of the forecast. So if the accumulation period is 6h, for
   the date 2020-01-01 00:00, the source will use the forecast [1]_ of
   2019-12-31 18:00 and the step 6h.

-  For ERA5, the data is accumulated since the last time step, and steps
   are hourly, and only available at 06Z and 18Z. So, a 6h accumulation
   for the date 2020-01-01 13:00, the source will use the forecast of
   2020-01-01 06:00 and the step 1-2h, 2-3h, 3-4h, 4-5h, 5-6h and 6-7h.

If the ``accumulation_period`` value is a pair of integers `[step1,
step2]`, the algorithm is different. The source will compute the
accumulation between the `step1` and `step2` previous forecasts that
validate at the given date at `step2`. For example, if the accumulation
period is `[6, 12]`, and the valid date is 2020-10-10 18:00, the source
will use the forecast of 2020-10-10 06:00 and the steps 6h and 12h.

Please note that ``accumulation_period=6`` and ``accumulation_period=[0,
6]`` are not equivalent. In the first case, the source can return an
accumulation between step 1h and step 7h if it is the most appropriate
data available, while in the second case, the source will always return
the accumulation between step 0h and step 6h, if available.

.. literalinclude:: yaml/accumulations1.yaml
   :language: yaml

or:

.. literalinclude:: yaml/accumulations2.yaml
   :language: yaml

.. [1]

   For ECMWF forecasts, the forecasts at 00Z and 12Z are from the stream
   `oper` while the forecasts at 06Z and 18Z are from the stream `scda`.
