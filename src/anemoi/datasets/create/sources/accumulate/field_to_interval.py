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

from anemoi.datasets.create.intervals import SignedInterval

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
    if endStep % 24 == 0:
        # Special case: endStep is exactly 24, 48, 72, etc.
        # Map to previous 24-hour boundary (24 -> 0, 48 -> 24, etc.)
        return endStep - 24, endStep

    # General case: floor to the nearest 24-hour boundary
    # (1-23 -> 0, 25-47 -> 24, etc.)
    return endStep - (endStep % 24), endStep


def _set_start_step_to_zero(startStep, endStep, field=None):
    # Because the data wrongly encode start_step, but end_step is correct
    return 0, endStep


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

        if startStep > endStep:
            startStep, endStep = endStep, startStep
        elif startStep == endStep:
            startStep, endStep = 0, endStep

        start_step = datetime.timedelta(hours=startStep)
        end_step = datetime.timedelta(hours=endStep)

        assert startStep >= 0, ("After patching, startStep must be >= 0", field, startStep, endStep)
        assert startStep < endStep, ("After patching, startStep must be < endStep", field, startStep, endStep)

        interval = SignedInterval(start=base_datetime + start_step, end=base_datetime + end_step, base=base_datetime)

        assert valid_date == interval.max, (valid_date, interval)

        return interval

    @staticmethod
    def _steps_and_validity(field, base_datetime: datetime.datetime) -> tuple[int, int, datetime.datetime]:
        """The field's ``(startStep, endStep, valid_date)`` in whole hours.

        GRIB-backed source fields carry the ``startStep``/``endStep``/
        ``validityDate`` keys directly. In-memory fields that carry only the
        earthkit time/proc *components* (e.g. those built with
        ``Field.from_numpy``) do not; there the window is derived from
        ``time.step`` and ``proc.time_value`` (``startStep = step − length``).

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
            return start_step, end_step, valid_date

        # In-memory field: derive the window from the earthkit time/proc components.
        step_hours = int(field.time.step().total_seconds() // 3600)
        length = field.proc.time_value()
        if length is None:
            raise ValueError(
                "accumulate: cannot determine the accumulation window of an in-memory field — "
                "it carries no GRIB 'startStep'/'endStep' and its earthkit 'proc.time_value' "
                "(the accumulation length) is None."
            )
        length_hours = int(length.total_seconds() // 3600)
        valid_date = base_datetime + datetime.timedelta(hours=step_hours)
        return step_hours - length_hours, step_hours, valid_date

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
