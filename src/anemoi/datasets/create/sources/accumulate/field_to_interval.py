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
from anemoi.datasets.create.intervals import step_to_timedelta
from anemoi.datasets.create.intervals import timedelta_to_step
from anemoi.datasets.create.intervals import step_to_timedelta

LOG = logging.getLogger(__name__)


def _set_start_step_from_end_step_ceiled_to_24_hours(startStep, endStep, field=None, steps=None):
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


def _set_start_step_to_zero(startStep, endStep, field=None, steps=None):
    # Because the data wrongly encode start_step, but end_step is correct
    return 0, endStep


COVERING_PATCH = "start_step_from_covering"


def _set_start_step_from_covering(startStep, endStep, field=None, steps=None):
    """Rebuild startStep from the intervals declared in ``covering:``.

    Some archives stamp ``startStep=0`` on every field even when the value is a
    per-interval statistic.  For instance ``rr``/``se-al-ec`` wind gusts at step
    9 hold the maximum over ``[6, 9]`` but are encoded as ``[0, 9]``.  ``endStep``
    is correct, so the declared covering says where the window really starts:
    the interval ending at 9 starts at 6.

    Only ``startStep`` is touched; a field whose ``endStep`` is not declared in
    ``covering:`` is an error rather than a guess.
    """
    if steps is None:
        raise ValueError(
            f"Patch {COVERING_PATCH!r} needs the step ranges declared in 'covering:', "
            "but none were bound to this FieldToInterval."
        )
    # `steps` is keyed by timedelta, and endStep may be an int or a sub-hourly
    # string ("10m"), so normalise before looking it up.
    key = step_to_timedelta(endStep)
    if key not in steps:
        declared = sorted(timedelta_to_step(k) for k in steps)
        raise ValueError(
            f"Patch {COVERING_PATCH!r}: 'covering:' declares no interval ending at step "
            f"{endStep} (declared end steps: {declared}). Field: {field}"
        )
    return steps[key], endStep


patch_registry = {
    "reset_24h_accumulations": _set_start_step_from_end_step_ceiled_to_24_hours,
    "set_start_step_to_zero": _set_start_step_to_zero,
    COVERING_PATCH: _set_start_step_from_covering,
}


class FieldToInterval:
    """Convert a field to its accumulation interval, applying patches if needed."""

    def __init__(
        self,
        patches: dict | None = None,
        steps: dict[datetime.timedelta, datetime.timedelta] | None = None,
        require_interval: bool = False,
    ):
        if patches is None:
            patches = []
        assert isinstance(patches, list), ("patches must be a list", patches)

        self.patches = patches
        for key in patches:
            if key not in patch_registry:
                raise ValueError(f"Unknown patch key: {key}")

        # end step -> start step, from the recipe's `covering:` declaration.
        # Only needed by COVERING_PATCH, and bound by the source once the covering is built.
        self.steps = steps

        # Reject fields that carry no interval instead of reading them as [0, endStep].
        # `accumulate` relies on that fallback (some archives encode accumulations with
        # startStep == endStep); `reduce` must not, or a snapshot would masquerade as a
        # window statistic.
        self.require_interval = require_interval

    @property
    def needs_declared_steps(self) -> bool:
        """Whether any requested patch needs the ``covering:`` step ranges."""
        return COVERING_PATCH in self.patches

    def bind_steps(self, steps: dict[datetime.timedelta, datetime.timedelta]) -> None:
        """Supply the end-step to start-step mapping declared in ``covering:``."""
        self.steps = steps

    def _reject_field_without_interval(self, field, endStep) -> None:
        """Raise on a field that covers a single instant rather than a window."""
        try:
            step_type = field.metadata("stepType")
        except Exception:
            step_type = "unknown"
        raise ValueError(
            f"Field {field} covers no time interval: startStep == endStep == {endStep} "
            f"(stepType={step_type!r}). This source combines fields that each cover a slice "
            "of the window, so every field must carry a real interval.\n"
            "- If this parameter is instantaneous (a snapshot, e.g. 2t or 10u), it cannot be "
            "aggregated this way. Averaging or maximising snapshots is a different operation "
            "from aggregating interval statistics, and is not supported.\n"
            "- If it really is an interval statistic whose metadata is wrong, repair startStep "
            "with a 'patch:' entry (see the accumulate documentation)."
        )

    def __call__(self, field) -> SignedInterval:
        date_str = str(field.metadata("date")).zfill(8)
        time_str = str(field.metadata("time")).zfill(4)
        base_datetime = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")

        endStep = field.metadata("endStep")
        startStep = field.metadata("startStep")

        LOG.debug(f" 🌧️:    field before patching: {startStep=}, {endStep=}")

        for patch_name in self.patches:
            patch_func = patch_registry[patch_name]
            startStep, endStep = patch_func(startStep, endStep, field, steps=self.steps)

        LOG.debug(f" 🌧️:    field after user patches: {startStep=}, {endStep=}")

        # Sub-hourly fields report their steps in minutes ("0m", "10m"), so parse
        # before comparing.
        start_step = step_to_timedelta(startStep)
        end_step = step_to_timedelta(endStep)

        if start_step > end_step:
            start_step, end_step = end_step, start_step
        elif start_step == end_step:
            if self.require_interval:
                self._reject_field_without_interval(field, endStep)
            start_step = datetime.timedelta(0)

        assert start_step >= datetime.timedelta(0), (
            "After patching, startStep must be >= 0",
            field,
            startStep,
            endStep,
        )
        assert start_step < end_step, ("After patching, startStep must be < endStep", field, startStep, endStep)

        interval = SignedInterval(start=base_datetime + start_step, end=base_datetime + end_step, base=base_datetime)

        date_str = str(field.metadata("validityDate")).zfill(8)
        time_str = str(field.metadata("validityTime")).zfill(4)
        valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        assert valid_date == interval.max, (valid_date, interval)

        return interval
