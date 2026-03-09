# (C) Copyright 2024 Anemoi contributors.
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

LOG = logging.getLogger(__name__)


def _parse_mars_times(raw_times) -> list[int]:
    """Convert a list of MARS time values to integer hours.

    Parameters
    ----------
    raw_times : list
        Time values as integers or ``HH:MM`` strings.

    Returns
    -------
    list of int
        Hour values as integers (e.g. ``0``, ``6``, ``12``, ``18``).

    Examples
    --------
    >>> _parse_mars_times([0])
    [0]
    >>> _parse_mars_times([0, 12])
    [0, 12]
    >>> _parse_mars_times(["00:00", "12:00"])
    [0, 12]
    >>> _parse_mars_times(["00:00", "12:00", "18:00"])
    [0, 12, 18]
    """
    if not isinstance(raw_times, (list, tuple)):
        raw_times = [raw_times]
    return [int(str(t).replace(":", "")) // 100 if ":" in str(t) else int(t) for t in raw_times]


def migrate_accumulations(config):
    """Migrate source 'accumulations' to the new 'accumulate' structure recursively."""
    if isinstance(config, dict):
        if "accumulations" in config:
            values = dict(config["accumulations"])
            accumulation_period = values.pop("accumulation_period", 6)
            if "step" in values:
                LOG.warning(
                    "Stripping 'step: %s' from accumulations source — "
                    "step is computed internally and any user-supplied value is ignored.",
                    values.pop("step"),
                )
            if "accumulations_reset_frequency" in values:
                LOG.warning(
                    "Stripping 'accumulations_reset_frequency: %s' from accumulations source — "
                    "this parameter has no equivalent in the new accumulate source.",
                    values.pop("accumulations_reset_frequency"),
                )
            if isinstance(accumulation_period, int):
                period = accumulation_period
                class_ = values.get("class", "od")
                stream = values.get("stream", "oper")
                if (class_, stream) == ("od", "enfo"):
                    # 'auto' raises NotImplementedError for od/enfo in the new code.
                    # Use explicit availability: accumulated-from-start, all four base times.
                    LOG.warning(
                        "'availability: auto' is not yet supported for class=od stream=enfo. "
                        "Using explicit availability with base times [0, 6, 12, 18]."
                    )
                    availability = [[bt, [f"0-{period}"]] for bt in [0, 6, 12, 18]]
                else:
                    availability = "auto"
            elif isinstance(accumulation_period, (list, tuple)):
                step1, step2 = accumulation_period
                if not isinstance(step1, int) or not isinstance(step2, int):
                    raise ValueError(f"Invalid accumulation_period: {accumulation_period}")
                period = step2 - step1
                steps = [f"0-{step2}"] if step1 == 0 else [f"0-{step1}", f"0-{step2}"]
                if "time" in values:
                    raw_times = values.pop("time")
                    base_times = _parse_mars_times(raw_times)
                    LOG.warning(
                        "Stripping 'time' from accumulations mars request — "
                        "using time values %s as availability base times.",
                        base_times,
                    )
                else:
                    base_times = [0, 6, 12, 18]
                availability = [[bt, steps] for bt in base_times]
            else:
                raise ValueError(f"Invalid accumulation_period: {accumulation_period}")
            if values.get("type") == "an":
                LOG.warning(
                    "Changing 'type: an' to 'type: fc' in accumulations mars source — "
                    "accumulated fields come from forecasts, not analyses."
                )
                values["type"] = "fc"
            result = {k: migrate_accumulations(v) for k, v in config.items() if k != "accumulations"}
            result["accumulate"] = {
                "period": period,
                "availability": availability,
                "source": {
                    "mars": values,
                },
            }
            return result
        return {k: migrate_accumulations(v) for k, v in config.items()}
    if isinstance(config, list):
        return [migrate_accumulations(item) for item in config]
    return config


def fix_datetimes(config):
    """Convert datetime objects to plain strings without T/Z notation.

    PyYAML parses timestamp strings into Python datetime objects at load time.
    When dumped back they become ISO 8601 (``2024-12-31T18:00:00Z``).
    This converts them back to simple strings (``2024-12-31 18:00:00``).
    """
    if isinstance(config, dict):
        return {k: fix_datetimes(v) for k, v in config.items()}
    if isinstance(config, list):
        return [fix_datetimes(item) for item in config]
    if isinstance(config, datetime.datetime):
        if config.hour == 0 and config.minute == 0 and config.second == 0:
            return config.strftime("%Y-%m-%d")
        return config.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(config, datetime.date):
        return config.strftime("%Y-%m-%d")
    return config


def remove_useless_common_block(config):
    """Remove 'common' keys from the config."""
    return {k: v for k, v in config.items() if k != "common"}


def migrate(config: dict) -> dict:
    config = fix_datetimes(config)
    config = migrate_accumulations(config)
    config = remove_useless_common_block(config)
    return config


def migrate_recipe(args: Any, config) -> None:

    print(f"Migrating {args.path}")

    migrated = migrate(config)

    if migrated == config:
        return None

    return migrated
