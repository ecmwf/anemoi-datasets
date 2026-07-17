# (C) Copyright 2024-2026 Anemoi contributors.
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
    """Migrate source 'accumulations' to the new 'accumulate' structure recursively.

    Recursively processes the config structure:
    - If a dict contains 'accumulations', converts it to new 'accumulate' format
    - Otherwise, recursively processes all nested dicts and lists
    - Returns primitive values unchanged (base case)
    """
    if isinstance(config, dict):
        if "accumulations" in config:
            values = dict(config["accumulations"])
            if "accumulation_period" not in values:
                LOG.warning(
                    "No 'accumulation_period' specified in accumulations source — " "using default value of 6 hours."
                )
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
        # No 'accumulations' key: recursively process nested structures
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


def migrate_allow_nans(config: dict) -> dict:
    allow_nans = config.get("statistics", {}).get("allow_nans")
    if allow_nans is not None:
        if isinstance(allow_nans, list):
            config = config.copy()
            config["statistics"] = config["statistics"].copy()
            config["statistics"]["allow_nans"] = bool(allow_nans)

    return config


def migrate_remapping(config: dict) -> dict:
    remapping = config.get("output", {}).get("remapping")
    if remapping is not None:
        config = config.copy()
        config["output"] = config["output"].copy()
        del config["output"]["remapping"]
        if not config["output"]:
            del config["output"]
    return config


def migrate_group_by(config: dict) -> dict:
    group_by = config.get("dates", {}).get("group_by")
    if group_by is not None:
        config = config.copy()
        group_by = config["dates"]["group_by"]
        del config["dates"]["group_by"]

        config.setdefault("build", {})
        config["build"] = config["build"].copy()
        config["build"].setdefault("group_by", group_by)

    return config


# ---------------------------------------------------------------------------
# accumulate: old spellings (availability/covering/accumulation) → new API
# ---------------------------------------------------------------------------


def _grid_to_steps(grid: list[int]) -> dict | list[dict]:
    """Convert a sorted step grid (hours) to Steps range dict(s)."""
    runs = []
    i, n = 0, len(grid)
    while i < n:
        if i + 1 < n:
            frequency = grid[i + 1] - grid[i]
            j = i + 1
            while j + 1 < n and grid[j + 1] - grid[j] == frequency:
                j += 1
            runs.append((grid[i], grid[j], frequency))
            i = j + 1
        else:
            value = grid[i]
            runs.append((value, value, value if value > 0 else 1))
            i += 1
    result = [{"start": f"{a}h", "end": f"{b}h", "frequency": f"{f}h"} for a, b, f in runs]
    return result[0] if len(result) == 1 else result


def _factorise_pairs(pairs) -> tuple[str, list[int]] | None:
    """Factorise raw (start, end) step pairs into (accumulated scheme, step grid)."""
    from math import gcd

    pairs = sorted({(int(a), int(b)) for a, b in pairs})

    if all(a == 0 for a, b in pairs):
        return "from-zero", sorted({b for _, b in pairs})

    if len({a for a, _ in pairs}) == len(pairs) and all(
        pairs[i][1] == pairs[i + 1][0] for i in range(len(pairs) - 1)
    ):
        return "from-previous-step", [pairs[0][0]] + [b for _, b in pairs]

    reset = 0
    for a, _ in pairs:
        reset = gcd(reset, a)
    if reset > 0 and all(a == (b - 1) // reset * reset for a, b in pairs):
        return f"from-zero-reset-every-{reset}h", sorted({b for _, b in pairs})

    return None


def _factorise_entries(entries, day_of_month=None) -> dict | None:
    """Factorise legacy (base_time, steps) entries into a from-trajectories payload."""
    from anemoi.datasets.create.sources.accumulate.interval_generators import normalise_steps

    groups: dict = {}
    for base_time, steps in entries:
        pairs = tuple(map(tuple, normalise_steps(steps)))
        groups.setdefault(pairs, []).append(base_time)
    if len(groups) != 1:
        return None
    pairs, base_times = next(iter(groups.items()))
    factorised = _factorise_pairs(pairs)
    if factorised is None:
        return None
    scheme, grid = factorised

    # times as integer hours: unquoted "HH:MM" strings do not survive a
    # YAML round-trip (YAML 1.1 parses 18:00 as the integer 1080)
    base_dates = {"times": _parse_mars_times(base_times)}
    if day_of_month is not None:
        base_dates["day_of_month"] = day_of_month
    return {"base_dates": base_dates, "steps": _grid_to_steps(grid), "accumulated": scheme}


def _convert_legacy_description(value) -> tuple[str, object] | None:
    """Convert a legacy availability/covering value to (description key, payload).

    Returns None when no faithful conversion exists (the caller keeps the
    old spelling and warns).
    """
    if value == "auto":
        return "from-trajectories", "auto"

    if isinstance(value, str):
        # frequency string: fixed-period increments (grib-index)
        import re

        if re.fullmatch(r"\d+\s*[a-zA-Z]*", value.strip()):
            return "from-increments", value
        return None

    if isinstance(value, dict):
        if len(value) == 1:
            key = next(iter(value))
            if key == "auto":
                return _convert_legacy_description(value["auto"])
            if key == "cycle":
                return "from-lookup-table", dict(value["cycle"])
            if key == "accumulated-from-start":
                return _convert_sugar("from-zero", value[key])
            if key == "accumulated-from-previous-step":
                return _convert_sugar("from-previous-step", value[key])

        if value.get("type") == "cycle":
            return "from-lookup-table", {k: v for k, v in value.items() if k != "type"}
        if value.get("type") == "accumulated-from-start":
            return _convert_sugar("from-zero", {k: v for k, v in value.items() if k != "type"})
        if value.get("type") == "accumulated-from-previous-step":
            return _convert_sugar("from-previous-step", {k: v for k, v in value.items() if k != "type"})

        if "mars" in value and len(value) == 1:
            from anemoi.datasets.create.sources.accumulate.description import _mars_archive_description

            mars = value["mars"]
            try:
                description = _mars_archive_description(mars.get("class"), mars.get("stream"), mars.get("origin"))
            except (ValueError, NotImplementedError):
                return None
            return "from-trajectories", description

        if "base_time" in value and "steps" in value:
            # legacy Pattern form
            base_time = value["base_time"]
            if base_time == "*":
                return None
            day_of_month = (value.get("base_date") or {}).get("day_of_month")
            if "search_range" in value:
                LOG.warning(
                    "Dropping 'search_range: %s' from accumulate availability — "
                    "the search reach is now derived from the archive description.",
                    value["search_range"],
                )
            description = _factorise_entries([(base_time, value["steps"])], day_of_month=day_of_month)
            if description is None:
                return None
            return "from-trajectories", description

        return None

    if isinstance(value, (list, tuple)):
        try:
            entries = [(base_time, steps) for base_time, steps in value]
        except (TypeError, ValueError):
            return None
        description = _factorise_entries(entries)
        if description is None:
            return None
        return "from-trajectories", description

    return None


def _convert_sugar(scheme: str, params: dict) -> tuple[str, dict] | None:
    """Convert the accumulated-from-start/-previous-step sugar to from-trajectories."""
    try:
        basetime = params["basetime"]
        frequency = int(params["frequency"])
        last_step = int(params["last_step"])
    except (KeyError, TypeError, ValueError):
        return None
    first = frequency if scheme == "from-zero" else 0
    return "from-trajectories", {
        "base_dates": {"times": _parse_mars_times(basetime)},
        "steps": {"start": f"{first}h", "end": f"{last_step}h", "frequency": f"{frequency}h"},
        "accumulated": scheme,
    }


def migrate_accumulate(config, trajectories: bool | None = None):
    """Rewrite accumulate blocks from the pre-redesign spellings to the new API.

    - ``accumulation:`` → ``accumulated:`` (trajectory recipes);
    - ``availability:`` / ``covering:`` → one of the description keys
      ``from-trajectories:`` / ``from-increments:`` / ``from-lookup-table:``
      (dropped instead in trajectory-layout recipes, where the old key was
      silently ignored and a description key is now an error).

    Legacy descriptions that cannot be factorised faithfully are left
    unchanged (as ``covering:``, still accepted with a DeprecationWarning).
    """
    if isinstance(config, dict):
        if trajectories is None:
            trajectories = (config.get("output") or {}).get("layout") == "trajectories"
        result = {}
        for k, v in config.items():
            if k == "accumulate" and isinstance(v, dict):
                result[k] = _migrate_accumulate_block(v, trajectories)
            else:
                result[k] = migrate_accumulate(v, trajectories)
        return result
    if isinstance(config, list):
        return [migrate_accumulate(item, trajectories) for item in config]
    return config


def _migrate_accumulate_block(block: dict, trajectories: bool) -> dict:
    result = {}
    for k, v in block.items():
        if k == "accumulation":
            result["accumulated"] = v
            continue
        if k in ("availability", "covering"):
            if trajectories:
                # The old code silently ignored the key in the trajectory
                # branch; the new API rejects a description there.
                LOG.warning(
                    "Dropping accumulate '%s: %s' — the trajectories layout imposes the "
                    "basetime, an archive description is not used.",
                    k,
                    v,
                )
                continue
            converted = _convert_legacy_description(v)
            if converted is None:
                LOG.warning(
                    "Cannot rewrite accumulate '%s: %s' as a description key "
                    "(from-trajectories/from-increments/from-lookup-table) — leaving it unchanged.",
                    k,
                    v,
                )
                result[k] = v
            else:
                new_key, payload = converted
                result[new_key] = payload
            continue
        result[k] = migrate_accumulate(v, trajectories)
    return result


def migrate(config: dict) -> dict:
    config = fix_datetimes(config)
    config = migrate_accumulations(config)
    config = migrate_accumulate(config)
    config = migrate_allow_nans(config)
    config = migrate_group_by(config)
    config = remove_useless_common_block(config)
    return config


def migrate_recipe(args: Any, config) -> None:

    LOG.info("Migrating %s", args.path)

    migrated = migrate(config)

    if migrated == config:
        return None

    return migrated
