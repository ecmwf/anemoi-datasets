# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def write_accumulated_field_with_valid_time(
    template, values, valid_date: datetime.datetime, period: datetime.timedelta, output
) -> Any:
    MISSING_VALUE = 1e-38
    assert np.all(values != MISSING_VALUE)

    date = (valid_date - period).strftime("%Y%m%d")
    time = (valid_date - period).strftime("%H%M")
    endStep = period

    hours = endStep.total_seconds() / 3600
    if hours != int(hours):
        raise ValueError(f"Accumulation period must be integer hours, got {hours}")
    hours = int(hours)

    if template.metadata("edition") == 1:
        # this is a special case for GRIB edition 1 which only supports integer hours up to 254
        assert hours <= 254, f"edition 1 accumulation period must be <=254 hours, got {hours}"
        output.write(
            values,
            template=template,
            date=int(date),
            time=int(time),
            stepType="instant",
            step=hours,
            check_nans=True,
            missing_value=MISSING_VALUE,
        )
    else:
        # this is the normal case for GRIB edition 2. And with edition 1 when hours are integer and <=254
        output.write(
            values,
            template=template,
            date=int(date),
            time=int(time),
            stepType="accum",
            startStep=0,
            endStep=hours,
            check_nans=True,
            missing_value=MISSING_VALUE,
        )


def write_accumulated_forecast_field(
    template,
    values,
    basetime: datetime.datetime,
    valid_date: datetime.datetime,
    period: datetime.timedelta,
    output,
) -> None:
    """Write an accumulated forecast field stamped with the basetime.

    Unlike :func:`write_accumulated_field_with_valid_time`, the output
    field's ``date``/``time`` keys are the model-run basetime (so the
    trajectory loader can recover ``(basetime, step)`` from metadata)
    and the step is the offset from the basetime to the validity time.

    Parameters
    ----------
    template
        Field used as GRIB template (header + grid).
    values
        Accumulated values array.
    basetime
        Model-run base time.
    valid_date
        Validity time at the end of the accumulation window.
    period
        Length of the accumulation window.
    output
        ``new_grib_output`` writer.
    """
    MISSING_VALUE = 1e-38
    assert np.all(values != MISSING_VALUE)

    end_step = (valid_date - basetime).total_seconds() / 3600
    start_step = (valid_date - basetime - period).total_seconds() / 3600
    if not (end_step.is_integer() and start_step.is_integer()):
        raise ValueError(f"Trajectory accumulation requires integer-hour steps; got start={start_step}, end={end_step}")
    end_step = int(end_step)
    start_step = int(start_step)

    date = int(basetime.strftime("%Y%m%d"))
    time = int(basetime.strftime("%H%M"))

    # Encode as an accumulation over [startStep, endStep] in the template's own
    # edition. GRIB1 stores this as timeRangeIndicator=4 with P1=startStep,
    # P2=endStep; GRIB2 as stepType=accum with startStep/endStep. Both keep the
    # lead time (``step`` == endStep) so the trajectory loader can recover
    # (basetime, step), while the accumulation period stays recoverable
    # downstream. Writing an *instant* field instead would collapse the window
    # and make the period unrecoverable.
    #
    # GRIB1 stores P1/P2 as single octets (<=255) counted in the hour time unit.
    # (Coarser units would extend the range but only for steps divisible by them;
    # we keep it simple and require hour-resolution offsets within the octet.)
    # Steps are integer hours here, so the binding limit is endStep <= 255 h
    # (~10.6 days). Fail early with a clear message rather than letting
    # output.write raise an opaque WrongStepError.
    if template.metadata("edition") == 1 and end_step > 255:
        raise ValueError(
            f"GRIB1 cannot encode an accumulation endStep of {end_step}h: P1/P2 are "
            "single octets limited to 255 h. We would need to use a GRIB2 template for these lead times."
            f"But we have the following template metadata: {template.metadata()}"
        )

    output.write(
        values,
        template=template,
        date=date,
        time=time,
        stepType="accum",
        startStep=start_step,
        endStep=end_step,
        check_nans=True,
        missing_value=MISSING_VALUE,
    )
