###############
 accumulations
###############

.. note::

   The `accumulated` source was previously named `accumulations` and
   using the `mars` source internally. This is no longer the case, and
   the `mars` source should be explicitly written in the recipe.

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

The source 'accumulate' needs two pieces of information to be able to
provide the accumulation over a given period: - The requested
accumulation period (such as 6h, 12h, etc). - The source of data, such
as 'mars', 'grib-index', etc. Currently only 'mars' and 'grib-index'
sources are implemented. - Optionally, hints about how the accumulation
is stored in the source dataset.

The `accumulate` source has some of that knowledge built-in.

Question: Should what was descibed be the default behavior (since the
beginnning over 6h)? If the source dataset is unknown, the package
assumes that the fields to use are accumulated since the beginning of
the forecast, over a 6h period.

this changed for now. the source 'mars' is known and in this cas the
package will analyse the request and decides wich intervals will be used
to reconstruct the requested accumulation interval. Depending on the
request (class, stream,...) the package may know how to reconstruct the
accumulation or not. or the source 'grib-index' We need to put 'hints' f
and other sources are not implemented yet. Requesting accumulation
periods strictly longer than 48h may lead to unexpected results,
depending on the source.

.. literalinclude:: yaml/accumulate-mars-1.yaml
   :language: yaml

The data used is accumulated from the beginning of the forecast, but the
most recent data available will be used:

-  For ECMWF operational forecasts, the data is accumulated from the
   beginning of the forecast. So if the accumulation period is 6h, for
   the date 2020-01-01 00:00, the source will use the forecast [1]_ of
   2019-12-31 18:00 and the step 6h.

-  For ERA5, the data is accumulated since the last time step, and steps
   are hourly, and only available at 06Z and 18Z. So, a 6h accumulation
   for the date 2020-01-01 13:00, the source will use the forecast of
   2020-01-01 06:00 and the step 1-2h, 2-3h, 3-4h, 4-5h, 5-6h and 6-7h.

When the package cannot deduce how to reconstruct the accumulation from
the source dataset, the user can provide the information with the
`data_accumulation_period` parameter or the `hints` parameter.

Using the `data_accumulation_period` parameter, the user can specify the
accumulation period used in the source dataset. This is useful when the
source dataset uses a different accumulation period than the one
requested and no ambiguity exists. For example, if the source dataset
contains 1h accumulations, the user can specify
`data_accumulation_period: 1h`.

.. literalinclude:: yaml/accumulate-grib-index.yaml
   :language: yaml

Note that even with this additional information, depending on the
metadata of the source dataset, the package may not be able to deduce
how to reconstruct the requested accumulation period.

When more detailed control is needed, the user can provide a description
of the accumulation intervals available in the source dataset with the
`hints` parameter. This parameter is a list of accumulation intervals
available in the source dataset. This `hints` parameter is experimental
and subject to change.

To request an accumulation between two forecast steps, the user can
provides hints as follows:

.. literalinclude:: yaml/accumulate-mars-2.yaml
   :language: yaml

The `hints` parameter describes, for each base date hour, the list of
accumulation intervals available in the source dataset. In the example
above, for base dates at 00Z, the available accumulation intervals are
[0-6] and [6-12], while for base dates at 12Z, the available
accumulation intervals are [12-18] and [18-24].

xxxxxxxxxxxxxxxxx

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
