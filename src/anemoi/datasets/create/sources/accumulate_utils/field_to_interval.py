# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging

from anemoi.utils.dates import frequency_to_timedelta

from .covering_intervals import SignedInterval

LOG = logging.getLogger(__name__)


class FieldToInterval:
    """Convert a field to its accumulation interval, applying patches if needed."""

    def __init__(self, patches: dict | None = None):
        if patches is None:
            patches = {}
        assert isinstance(patches, dict), ("patches must be a dict", patches)

        self.patches = patches
        for key in patches:
            if key not in (
                "start_step_is_zero",
                "start_step_is_end_step",
                "start_step_greater_than_end_step",
            ):
                raise ValueError(f"Unknown patch key: {key}")

    def __call__(self, field) -> SignedInterval:
        date_str = str(field.metadata("date")).zfill(8)
        time_str = str(field.metadata("time")).zfill(4)
        base_datetime = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")

        endStep = field.metadata("endStep")
        startStep = field.metadata("startStep")

        LOG.debug(f"    field before patching: {startStep=}, {endStep=}")

        if startStep > endStep:
            startStep, endStep = self.start_step_greater_than_end_step(startStep, endStep, field=field)
        elif startStep == endStep:
            startStep, endStep = self.start_step_is_end_step(startStep, endStep, field=field)
        elif frequency_to_timedelta(startStep).total_seconds() == 0:
            startStep, endStep = self.start_step_is_zero(startStep, endStep, field=field)

        LOG.debug(f"    field after patching : {startStep=}, {endStep=}")

        start_step = datetime.timedelta(hours=startStep)
        end_step = datetime.timedelta(hours=endStep)

        assert startStep >= 0, ("After patching, startStep must be >= 0", field, startStep, endStep)
        assert startStep < endStep, ("After patching, startStep must be < endStep", field, startStep, endStep)

        interval = SignedInterval(start=base_datetime + start_step, end=base_datetime + end_step, base=base_datetime)

        date_str = str(field.metadata("validityDate")).zfill(8)
        time_str = str(field.metadata("validityTime")).zfill(4)
        valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        assert valid_date == interval.max, (valid_date, interval)

        return interval

    def start_step_is_zero(self, startStep, endStep, field=None):
        # Patch to handle cases where start_step is zero
        # No patch yet implemented
        match self.patches.get("start_step_is_zero", None):
            case False | None:
                pass  # do nothing
            case _ as options:
                raise ValueError(f"Unknown option for patch.start_step_is_zero: {options}")

        return startStep, endStep

    def start_step_is_end_step(self, startStep, endStep, field=None):
        # Patch to handle cases where start_step equals end_step
        # this should not happen in normal cases but some datasets have this issue
        # The default is to set start_step to zero
        # This can be disabled by setting the patch to False

        match self.patches.get("start_step_is_end_step", "set_start_step_to_zero"):
            case False | None:
                pass  # do nothing

            case "set_from_end_step_ceiled_to_24_hours":
                startStep, endStep = _set_start_step_from_end_step_ceiled_to_24_hours(startStep, endStep, field=field)

            case "set_start_step_to_zero":
                startStep, endStep = 0, endStep

            case _ as options:
                raise ValueError(f"Unknown option for patch.start_step_is_end_step: {options}")

        return startStep, endStep

    def start_step_greater_than_end_step(self, startStep, endStep, field=None):

        # Patch to handle cases where start_step is greater than end_step
        # this should not happen in normal cases but some datasets have this issue
        # The default is to do swap the values of start_step and end_step
        # This can be disabled by setting the patch to False

        match self.patches.get("start_step_greater_than_end_step", None):

            case False | None:
                pass  # do nothing

            case "swap":
                startStep, endStep = endStep, startStep

            case _ as options:
                raise ValueError(f"Unknown option for patch.start_step_greater_than_end_step: {options}")

        return startStep, endStep


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
