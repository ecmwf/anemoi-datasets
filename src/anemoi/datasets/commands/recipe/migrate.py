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
import warnings
from typing import Any

from anemoi.datasets.create.intervals import timedelta_to_step

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


def _frequency_string(value: datetime.timedelta) -> str:
    """Format a lead time as a recipe frequency string.

    Whole hours are written in hours (``"24h"``, not ``"1d"``, which is what
    ``frequency_to_string`` would give) so that migrating an hour-based recipe
    reproduces the spelling it has always had; anything finer is written in
    minutes.  Both forms are accepted by ``frequency_to_timedelta``, which
    does not understand the compound ``"1h10m"`` archive-step syntax.
    """
    seconds = int(value.total_seconds())
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def _grid_to_steps(grid: list[datetime.timedelta], frequency: datetime.timedelta | None = None) -> dict | list[dict]:
    """Convert a sorted step grid to Steps range dict(s).

    ``frequency`` is the known spacing, when the scheme dictates one
    (from-previous-step): an isolated grid value carries no spacing of its
    own, and guessing it would mis-describe the accumulation length.
    """
    zero = datetime.timedelta(0)
    runs = []
    i, n = 0, len(grid)
    while i < n:
        if i + 1 < n:
            spacing = grid[i + 1] - grid[i]
            j = i + 1
            while j + 1 < n and grid[j + 1] - grid[j] == spacing:
                j += 1
            runs.append((grid[i], grid[j], spacing))
            i = j + 1
        else:
            value = grid[i]
            runs.append((value, value, frequency or (value if value > zero else datetime.timedelta(hours=1))))
            i += 1
    result = [
        {
            "start": _frequency_string(a),
            "end": _frequency_string(b),
            "frequency": _frequency_string(f),
        }
        for a, b, f in runs
    ]
    return result[0] if len(result) == 1 else result


def _factorise_pairs(
    pairs,
) -> tuple[str, list[datetime.timedelta], datetime.timedelta | None] | None:
    """Factorise raw (start, end) step pairs into (accumulation scheme, step grid, frequency).

    The pairs are lead-time timedeltas, so a sub-hourly archive factorises
    just like an hourly one.
    """
    from math import gcd

    zero = datetime.timedelta(0)
    pairs = sorted({(a, b) for a, b in pairs})

    if any(b <= a for a, b in pairs):
        # Degenerate legacy entries ("6-6", "6-0") name no real field; any
        # factorisation would invent one.
        return None

    if all(a == zero for a, b in pairs):
        return "from-zero", sorted({b for _, b in pairs if b > zero}), None

    if len({a for a, _ in pairs}) == len(pairs) and all(pairs[i][1] == pairs[i + 1][0] for i in range(len(pairs) - 1)):
        # `steps` lists the steps at which fields exist; the accumulation
        # length is the spacing, so it has to be regular to be expressible.
        # It is now stated as the duration itself (the former from-previous-step).
        lengths = {b - a for a, b in pairs}
        if len(lengths) != 1:
            return None
        length = next(iter(lengths))
        return _frequency_string(length), [b for _, b in pairs], length

    reset_seconds = 0
    for a, _ in pairs:
        reset_seconds = gcd(reset_seconds, int(a.total_seconds()))
    if reset_seconds > 0:
        reset = datetime.timedelta(seconds=reset_seconds)
        # The reset boundary is the largest multiple of `reset` strictly below b.
        if all(a == (b - datetime.timedelta.resolution) // reset * reset for a, b in pairs):
            return (
                f"from-zero-reset-every-{_frequency_string(reset)}",
                sorted({b for _, b in pairs}),
                None,
            )

    return None


def _factorise_entries(entries, day_of_month=None) -> dict | None:
    """Factorise legacy (base_time, steps) entries into a ``base_dates`` / ``steps`` payload."""
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
    scheme, grid, frequency = factorised
    if not grid:
        return None

    # times as integer hours: unquoted "HH:MM" strings do not survive a
    # YAML round-trip (YAML 1.1 parses 18:00 as the integer 1080)
    base_dates = {"times": sorted(set(_parse_mars_times(base_times)))}
    if day_of_month is not None:
        base_dates["day_of_month"] = day_of_month

    steps_repr = _grid_to_steps(grid, frequency)
    if isinstance(steps_repr, dict):
        # A regular grid: a compact range + the accumulation scheme.
        return {"base_dates": base_dates, "steps": steps_repr, "accumulation": scheme}
    # An irregular grid: the explicit (start, end) pairs are the whole
    # description, so no 'accumulation' is written (they cannot be a range).
    ordered = sorted({(a, b) for a, b in pairs}, key=lambda p: (p[1], p[0]))
    return {
        "base_dates": base_dates,
        "steps": [f"{timedelta_to_step(a)}-{timedelta_to_step(b)}" for a, b in ordered],
    }


def _convert_legacy_description(value) -> tuple[str, object] | None:
    """Convert a legacy availability/covering value to an internal (``kind``, payload) pair.

    Returns None when no faithful conversion exists (the caller keeps the
    old spelling and warns).
    """
    if value == "auto":
        return "trajectories", "auto"

    if isinstance(value, str):
        # frequency string: fixed-period increments (grib-index)
        import re

        if re.fullmatch(r"\d+\s*[a-zA-Z]*", value.strip()):
            return "valid-time", value
        return None

    if isinstance(value, dict):
        if len(value) == 1:
            key = next(iter(value))
            if key == "auto":
                return _convert_legacy_description(value["auto"])
            if key == "cycle":
                return "lookup-table", dict(value["cycle"])
            if key == "accumulated-from-start":
                return _convert_sugar("from-zero", value[key])
            if key == "accumulated-from-previous-step":
                return _convert_sugar("from-previous-step", value[key])

        if value.get("type") == "cycle":
            return "lookup-table", {k: v for k, v in value.items() if k != "type"}
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
            return "trajectories", description

        if "base_time" in value and "steps" in value:
            # legacy Pattern form
            base_time = value["base_time"]
            if base_time == "*":
                return None
            base_date = value.get("base_date") or {}
            if set(base_date) - {"day_of_month"}:
                # An unknown base_date selector must not be silently dropped.
                return None
            day_of_month = base_date.get("day_of_month")
            if "search_range" in value:
                LOG.warning(
                    "Dropping 'search_range: %s' from accumulate availability — "
                    "the search reach is now derived from the source-data description.",
                    value["search_range"],
                )
            description = _factorise_entries([(base_time, value["steps"])], day_of_month=day_of_month)
            if description is None:
                return None
            return "trajectories", description

        return None

    if isinstance(value, (list, tuple)):
        try:
            entries = [(base_time, steps) for base_time, steps in value]
        except (TypeError, ValueError):
            return None
        description = _factorise_entries(entries)
        if description is None:
            return None
        return "trajectories", description

    return None


def _convert_sugar(scheme: str, params: dict) -> tuple[str, dict] | None:
    """Convert the accumulated-from-start/-previous-step sugar to a trajectories payload."""
    try:
        basetime = params["basetime"]
        frequency = int(params["frequency"])
        last_step = int(params["last_step"])
    except (KeyError, TypeError, ValueError):
        return None
    # `steps` lists the steps at which fields exist: the first is one
    # accumulation length into the forecast, under either scheme.  A
    # per-step scheme is now stated as the duration itself.
    first = frequency
    accumulation = f"{frequency}h" if scheme == "from-previous-step" else scheme
    return "trajectories", {
        "base_dates": {"times": _parse_mars_times(basetime)},
        "steps": {"start": f"{first}h", "end": f"{last_step}h", "frequency": f"{frequency}h"},
        "accumulation": accumulation,
    }


#: Sentinel: this description is the default, so the key is dropped entirely.
_OMIT = object()


def _to_from(kind: str, payload) -> dict | str | object:
    """Convert an internal ``(kind, payload)`` pair to a structural ``from:`` block.

    There is no ``type:`` key: the ``from:`` shape is recognised
    structurally — ``base_dates``/``steps`` for trajectories,
    ``lookup-table`` for the table, a bare ``accumulation`` otherwise.
    Returns :data:`_OMIT` when the result would be ``auto`` (the default),
    so the migrated recipe simply carries no ``from:``.
    """
    if payload == "auto":
        return _OMIT
    if kind == "valid-time":
        return {"accumulation": payload}
    if kind == "lookup-table":
        return {"lookup-table": dict(payload)}
    # trajectories: base_dates / steps / accumulation, no discriminator.
    return dict(payload)


def migrate_accumulate(config, trajectories: bool | None = None):
    """Rewrite accumulate blocks from the pre-redesign spellings to the ``from:`` API.

    - block-level ``accumulation:`` → a bare ``from: {accumulation: ...}``,
      or, in a trajectory-layout recipe, the ``from-layout`` form
      ``from: {base_dates: from-layout, steps: from-layout, accumulation: ...}``;
    - ``availability:`` / ``covering:`` → a structural ``from:`` block (dropped
      instead in trajectory-layout recipes, where the old key was silently
      ignored and the run grid is imposed by the layout).

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
    """Convert one accumulate block, falling back to the unchanged block.

    Conversion is attempted by :func:`_convert_accumulate_block`; if it
    raises, or if the converted block does not pass the new-API schema,
    the original block is returned untouched (its deprecated spellings are
    still accepted at run time, with a warning) — a wrong migration must
    never be silently produced.
    """
    legacy = [k for k in ("availability", "covering", "accumulation") if k in block]
    if ("from" in block and legacy) or ("availability" in block and "covering" in block):
        # The block is over-specified; validation rejects it as written and
        # any rewrite would silently pick a winner. Surface, don't guess.
        keys = (["from"] if "from" in block else []) + legacy
        LOG.warning(
            "accumulate block carries more than one source-data description (%s) — "
            "not migrating it; remove the extra key(s) first.",
            ", ".join(keys),
        )
        return block

    try:
        result = _convert_accumulate_block(block, trajectories)
    except Exception as e:
        LOG.warning("Cannot rewrite accumulate block (%s: %s) — leaving it unchanged.", type(e).__name__, e)
        return block

    if result != block:
        from anemoi.datasets.create.sources.accumulate.description import AccumulateSchema
        from anemoi.datasets.create.sources.accumulate.description import FromBare
        from anemoi.datasets.create.sources.accumulate.description import check_valid_time_source

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                schema = AccumulateSchema.model_validate(result)
            # The description-level rules for a bare `from:` live in the `Recipe`
            # model rather than the block schema; apply them here too, so a block
            # the recipe would reject is left unchanged instead of silently
            # rewritten into an invalid one.
            if isinstance(schema.from_, FromBare):
                check_valid_time_source(schema.from_, period=schema.period)
        except Exception as e:
            LOG.warning(
                "Migrated accumulate block does not validate (%s) — leaving the block unchanged.",
                str(e).strip().splitlines()[-1],
            )
            return block

    return result


def _convert_accumulate_block(block: dict, trajectories: bool) -> dict:
    result = {}
    for k, v in block.items():
        if k == "accumulation":
            if not trajectories:
                # The old code only read the block-level scheme key in the
                # trajectory branch; anywhere else it was dead weight.
                LOG.warning(
                    "Dropping accumulate 'accumulation: %s' — it was only used in " "'layout: trajectories' recipes.",
                    v,
                )
                continue
            # The block-level scheme key described the run the trajectory
            # layout imposes; it moves inside `from:` as the `from-layout`
            # sentinel (on both base_dates and steps) plus the scheme.
            result["from"] = {"base_dates": "from-layout", "steps": "from-layout", "accumulation": v}
            continue
        if k in ("availability", "covering"):
            if trajectories:
                # The old code silently ignored the key in the trajectory
                # branch; the new API rejects a description there.
                LOG.warning(
                    "Dropping accumulate '%s: %s' — the trajectories layout imposes the "
                    "basetime, a source-data description is not used.",
                    k,
                    v,
                )
                continue
            converted = _convert_legacy_description(v)
            if converted is None:
                LOG.warning(
                    "Cannot rewrite accumulate '%s: %s' as a 'from:' description — leaving it unchanged.",
                    k,
                    v,
                )
                result[k] = v
            else:
                kind, payload = converted
                as_from = _to_from(kind, payload)
                if as_from is not _OMIT:
                    result["from"] = as_from
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
