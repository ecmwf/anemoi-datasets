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

from .covering_intervals import SignedInterval

LOG = logging.getLogger(__name__)


class FieldToInterval:
    """Convert a field to its accumulation interval, applying patches if needed."""

    def __init__(self, patches: list[dict] | dict):
        if isinstance(patches, dict):
            patches = [patches]
        for patch in patches:
            assert isinstance(patch, dict), f"Each patch must be a dict, got {type(patch)}"
            if len(patch) != 1:
                raise ValueError(f"Each patch must have exactly one key, got {patch}")
            if not hasattr(self, next(iter(patch.keys()))):
                raise ValueError(f"Unknown patch method: {next(iter(patch.keys()))}")
        self.patches = patches

    def __call__(self, field) -> SignedInterval:
        date_str = str(field.metadata("date")).zfill(8)
        time_str = str(field.metadata("time")).zfill(4)
        base_datetime = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")

        endStep = field.metadata("endStep")
        startStep = field.metadata("startStep")

        LOG.debug(f"    field before patching: {startStep=}, {endStep=}")
        for patch in self.patches:
            LOG.debug(f"      applying patch {patch}")
            name, options = next(iter(patch.items()))
            if options is False or options is None:
                continue
            startStep, endStep = getattr(self, name)(options, startStep, endStep, field=field)
            LOG.debug(f"        after patch {name}: {startStep=}, {endStep=}")
        LOG.debug(f"    field after patching : {startStep=}, {endStep=}")

        start_step = datetime.timedelta(hours=startStep)
        end_step = datetime.timedelta(hours=endStep)

        interval = SignedInterval(start=base_datetime + start_step, end=base_datetime + end_step, base=base_datetime)

        date_str = str(field.metadata("validityDate")).zfill(8)
        time_str = str(field.metadata("validityTime")).zfill(4)
        valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        assert valid_date == interval.max, (valid_date, interval)

        return interval

    def start_step_is_zero(self, options, startStep, endStep, field=None):
        if startStep != 0:
            raise ValueError(
                f"startStep must be 0 to apply 'start_step_is_zero' patch, got {startStep} for field {field}"
            )

        match options:
            case _:
                raise ValueError(f"Unknown option for patch.start_step_is_zero: {options}")

        return startStep, endStep

    def start_step_is_end_step(self, options, startStep, endStep, field=None):
        if startStep != endStep:
            raise ValueError(
                f"startStep must be equal to endStep to apply 'start_step_is_end_step' patch, got {startStep} != {endStep} for field {field}"
            )

        match options:
            case "set_from_end_step_ceiled_to_24_hours":
                startStep, endStep = _set_start_step_from_end_step_ceiled_to_24_hours(
                    options, startStep, endStep, field=field
                )
            case 0:
                startStep, endStep = 0, endStep
            case _:
                raise ValueError(f"Unknown option for patch.start_step_is_end_step: {options}")

        return startStep, endStep

    def start_step_greater_than_end_step(self, options, startStep, endStep, field=None):
        if startStep < endStep:
            raise ValueError(f"startStep expected to be >= endStep: got {startStep} < {endStep} for field {field}")

        match options:
            case "swap":
                startStep, endStep = endStep, startStep
            case _:
                raise ValueError(f"Unknown option for patch.start_step_greater_than_end_step: {options}")

        return startStep, endStep


def _set_start_step_from_end_step_ceiled_to_24_hours(options, startStep, endStep, field=None):
    # because the data wrongly encode start_step, but end_step is correct
    # and we know that accumulations are always reseted every multiple of 24 hours
    #
    # 1-1 -> 0-1
    # 2-2 -> 0-2
    # ...
    # 24-24 -> 0-24
    # 25-25 -> 24-25
    # 26-26 -> 24-26
    # ...
    # 48-48 -> 24-48
    #
    if endStep % 24 == 0:
        # Special case: endStep is exactly 24, 48, 72, etc.
        # Map to previous 24-hour boundary (24 -> 0, 48 -> 24, etc.)
        startStep = endStep - 24
    else:
        # General case: floor to the nearest 24-hour boundary
        # (1-23 -> 0, 25-47 -> 24, etc.)
        startStep = endStep - (endStep % 24)

    assert startStep >= 0, ("After patching, startStep must be >= 0", field, startStep, endStep)
    assert startStep < endStep, ("After patching, startStep must be < endStep", field, startStep, endStep)

    return startStep, endStep
