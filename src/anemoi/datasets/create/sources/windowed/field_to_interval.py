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

from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.intervals import step_to_timedelta

LOG = logging.getLogger(__name__)


def _set_start_step_from_end_step_ceiled_to_24_hours(startStep, endStep, field=None):
    # Because the data wrongly encode start_step, but end_step is correct
    # and we know that accumulations are always reseted every multiple of 24 hours
    #
    # 1-1 -> 0-1
    # 2-2 -> 0-2
    # ...
    # 23-23 -> 0-23
    # 24-24 -> 0-24
    # 25-25 -> 24-25
    # 26-26 -> 24-26
    # ...
    # 47-47 -> 24-47
    # 48-48 -> 24-48
    # 49-49 -> 48-49
    # 50-50 -> 48-50
    # etc.
    #
    # The steps are timedeltas, so a sub-hourly step is placed inside the
    # right 24-hour window rather than truncated to the hour.
    day = datetime.timedelta(hours=24)
    if not endStep % day:
        # Special case: endStep is exactly 24, 48, 72, etc.
        # Map to previous 24-hour boundary (24 -> 0, 48 -> 24, etc.)
        return endStep - day, endStep

    # General case: floor to the nearest 24-hour boundary
    # (1-23 -> 0, 25-47 -> 24, etc.)
    return endStep - (endStep % day), endStep


def _set_start_step_to_zero(startStep, endStep, field=None):
    # Because the data wrongly encode start_step, but end_step is correct
    return datetime.timedelta(0), endStep


#: ``indicatorOfUnitOfTimeRange`` codes finer than an hour (GRIB code table 4.4).
_SUB_HOURLY_UNITS = {0: "minutes", 13: "seconds", 14: "15 minutes", 15: "30 minutes"}


patch_registry = {
    "reset_24h_accumulations": _set_start_step_from_end_step_ceiled_to_24_hours,
    "set_start_step_to_zero": _set_start_step_to_zero,
}


class FieldToInterval:
    """Convert a field to its accumulation interval, applying patches if needed."""

    def __init__(self, patches: dict | None = None):
        if patches is None:
            patches = []
        assert isinstance(patches, list), ("patches must be a list", patches)

        self.patches = patches
        for key in patches:
            if key not in patch_registry:
                raise ValueError(f"Unknown patch key: {key}")

    def __call__(self, field) -> SignedInterval:
        base_datetime = self._base_datetime(field)

        startStep, endStep, valid_date = self._steps_and_validity(field, base_datetime)

        LOG.debug(f" 🌧️:    field before patching: {startStep=}, {endStep=}")

        for patch_name in self.patches:
            patch_func = patch_registry[patch_name]
            startStep, endStep = patch_func(startStep, endStep, field)

        LOG.debug(f" 🌧️:    field after user patches: {startStep=}, {endStep=}")

        start_step, end_step = startStep, endStep

        if start_step > end_step:
            start_step, end_step = end_step, start_step
        elif start_step == end_step:
            start_step = datetime.timedelta(0)

        assert start_step >= datetime.timedelta(0), (
            "After patching, startStep must be >= 0",
            field,
            startStep,
            endStep,
        )
        assert start_step < end_step, ("After patching, startStep must be < endStep", field, startStep, endStep)

        interval = SignedInterval(start=base_datetime + start_step, end=base_datetime + end_step, base=base_datetime)

        assert valid_date == interval.max, (valid_date, interval)

        return interval

    @staticmethod
    def _steps_and_validity(
        field, base_datetime: datetime.datetime
    ) -> tuple[datetime.timedelta, datetime.timedelta, datetime.datetime]:
        """The field's ``(startStep, endStep, valid_date)``, the steps as timedeltas.

        GRIB-backed source fields carry the ``startStep``/``endStep``/
        ``validityDate`` keys directly. In-memory fields that carry only the
        earthkit time/proc *components* (e.g. those built with
        ``Field.from_numpy``) do not; there the window is derived from
        ``time.step`` and ``proc.time_value`` (``startStep = step − length``).

        The GRIB keys are returned in the units of the message's
        ``stepUnits``, and a minute-unit field reads back as the *string*
        ``"0m"`` / ``"10m"`` rather than a number, so both are parsed with
        ``step_to_timedelta`` (a bare number keeps meaning hours).

        Only the *absence* of the GRIB step keys selects the in-memory path
        (a missing key raises ``KeyError``/``AttributeError``); any other error
        — including a malformed ``validityDate`` — is left to propagate rather
        than being silently rerouted.
        """
        end_step = FieldToInterval._optional_metadata(field, "endStep")
        if end_step is not None:
            start_step = field.metadata("startStep")
            date_str = str(field.metadata("validityDate")).zfill(8)
            time_str = str(field.metadata("validityTime")).zfill(4)
            valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
            start, end = step_to_timedelta(start_step), step_to_timedelta(end_step)
            FieldToInterval._check_step_unit(field, base_datetime, valid_date, end, end_step)
            return start, end, valid_date

        # In-memory field: derive the window from the earthkit time/proc
        # components, which are timedeltas (so sub-hourly windows survive).
        step = field.time.step()
        length = field.proc.time_value()
        if length is None:
            raise ValueError(
                "accumulate: cannot determine the accumulation window of an in-memory field — "
                "it carries no GRIB 'startStep'/'endStep' and its earthkit 'proc.time_value' "
                "(the accumulation length) is None."
            )
        valid_date = base_datetime + step
        return step - length, step, valid_date

    @staticmethod
    def _check_step_unit(field, base_datetime, valid_date, end_step, raw_end_step) -> None:
        """Refuse a step whose unit cannot be trusted, instead of mis-reading it.

        The validity time is unambiguous, so it says what the end step must be.
        A GRIB2 message with sub-hourly steps spells them out (``"10m"``) and
        agrees; a GRIB **edition 1** message cannot represent such a step in
        the default hour view, and eccodes then hands back the raw number
        (``10`` for 10 minutes), which reads as 10 *hours*.  Rather than build
        an interval an order of magnitude too long — and fail later with an
        opaque assertion — say what is wrong.
        """
        if base_datetime + end_step == valid_date:
            return

        unit = FieldToInterval._optional_metadata(field, "indicatorOfUnitOfTimeRange")
        if unit in _SUB_HOURLY_UNITS:
            raise ValueError(
                f"accumulate: this field's step ({raw_end_step!r}) is encoded in "
                f"{_SUB_HOURLY_UNITS[unit]} but its validity time says the step is "
                f"{frequency_to_string(valid_date - base_datetime)}. GRIB edition "
                f"{FieldToInterval._optional_metadata(field, 'edition')} cannot represent a "
                "sub-hourly step, so the value read back cannot be trusted; re-encode the "
                "archive as GRIB edition 2, whose steps carry their own unit ('10m')."
            )
        # Any other disagreement is left to the caller's own assertion, which
        # reports the whole interval (patches may legitimately move startStep,
        # but never the end of the window).

    @staticmethod
    def _optional_metadata(field, key: str):
        """Return the GRIB metadata *key*, or ``None`` when the field does not carry it.

        A GRIB-backed field exposes keys such as ``endStep``; an in-memory field
        does not and raises ``KeyError``/``AttributeError`` — reported here as
        ``None`` so the caller can take the in-memory path. Any other exception
        is not caught.
        """
        try:
            return field.metadata(key)
        except (KeyError, AttributeError):
            return None

    @staticmethod
    def _base_datetime(field) -> datetime.datetime:
        """The model-run base time of *field*.

        The mars 'date'/'time' metadata keys give the base time for ordinary
        forecasts, but for hindcast fields (eefh/enfh) 'date' is the reforecast
        reference date while the run actually starts at hdate (= dataDate); the
        field's time component resolves the correct base time in both cases.
        """
        try:
            return field.time.base_datetime()
        except AttributeError:
            date_str = str(field.metadata("date")).zfill(8)
            time_str = str(field.metadata("time")).zfill(4)
            return datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
