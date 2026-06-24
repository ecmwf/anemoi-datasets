# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Any

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_field_with_valid_datetime
from numpy.typing import NDArray


def build_accumulated_field_with_valid_time(
    template: Any, values: NDArray, valid_date: datetime.datetime, period: datetime.timedelta
) -> Any:
    """Build an in-memory accumulated field stamped with its validity time.

    The field's ``date``/``time`` keys are the window start (``valid_date −
    period``) and the step is the accumulation period, so the field's valid time
    is ``valid_date``. Used by the archive (validity-date) accumulation path.

    Parameters
    ----------
    template
        Field providing the grid and the param/level/number metadata.
    values
        Accumulated values array.
    valid_date
        Validity time at the end of the accumulation window.
    period
        Length of the accumulation window.

    Returns
    -------
    Any
        A new field carrying *values*, the template's geography, and the
        accumulation metadata.
    """
    base = valid_date - period
    hours = period.total_seconds() / 3600
    if hours != int(hours):
        raise ValueError(f"Accumulation period must be integer hours, got {hours}")
    hours = int(hours)

    field = new_field_from_numpy(
        values,
        template=template,
        date=int(base.strftime("%Y%m%d")),
        time=int(base.strftime("%H%M")),
        step=hours,
        startStep=0,
        endStep=hours,
        stepType="accum",
        # The cube builder reads ``stepTypeForConversion`` (not ``stepType``) to
        # classify the field as an accumulation over [startStep, endStep].
        stepTypeForConversion="accum",
    )
    return new_field_with_valid_datetime(field, valid_date)


def build_accumulated_forecast_field(
    template: Any,
    values: NDArray,
    basetime: datetime.datetime,
    valid_date: datetime.datetime,
    period: datetime.timedelta,
) -> Any:
    """Build an in-memory accumulated forecast field stamped with the basetime.

    Unlike :func:`build_accumulated_field_with_valid_time`, the output field's
    ``date``/``time`` keys are the model-run basetime (so the trajectory loader
    can recover ``(basetime, step)`` from metadata) and the step is the offset
    from the basetime to the validity time.

    Parameters
    ----------
    template
        Field providing the grid and the param/level/number metadata.
    values
        Accumulated values array.
    basetime
        Model-run base time.
    valid_date
        Validity time at the end of the accumulation window.
    period
        Length of the accumulation window.

    Returns
    -------
    Any
        A new field carrying *values*, the template's geography, and the
        forecast accumulation metadata.
    """
    end_step = (valid_date - basetime).total_seconds() / 3600
    start_step = (valid_date - basetime - period).total_seconds() / 3600
    if not (end_step.is_integer() and start_step.is_integer()):
        raise ValueError(f"Trajectory accumulation requires integer-hour steps; got start={start_step}, end={end_step}")
    end_step = int(end_step)
    start_step = int(start_step)

    # The forecast convention keeps ``date``/``time`` at the basetime and ``step``
    # at the lead time, so the trajectory loader can recover ``(basetime, step)``.
    # ``valid_datetime`` is set as a plain metadata override (rather than via
    # ``new_field_with_valid_datetime``, which would rewrite date/time/step to the
    # analysis convention, i.e. date/time = validity time and step = 0).
    return new_field_from_numpy(
        values,
        template=template,
        date=int(basetime.strftime("%Y%m%d")),
        time=int(basetime.strftime("%H%M")),
        step=end_step,
        startStep=start_step,
        endStep=end_step,
        stepType="accum",
        # The cube builder reads ``stepTypeForConversion`` (not ``stepType``) to
        # classify the field as an accumulation over [startStep, endStep].
        stepTypeForConversion="accum",
        valid_datetime=valid_date,
    )
