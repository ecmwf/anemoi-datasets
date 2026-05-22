# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import argparse
import re
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Iterable

import ruamel.yaml
from ruamel.yaml.constructor import RoundTripConstructor

from . import Command

# Keys that indicate a dict is a MARS-like request.
MARS_KEYS = {"class", "expver", "stream", "levtype", "param", "type"}

# Level keys to reduce.
LEVEL_KEYS = {"levelist", "level"}


class LenientConstructor(RoundTripConstructor):
    """Like RoundTripConstructor but tolerates duplicate YAML merge keys (<<)."""

    def flatten_mapping(self, node):
        # Deduplicate << merge keys: keep only the first, drop the rest.
        # The YAML spec technically forbids this but some recipes use it.
        seen_merge = False
        new_value = []
        for key_node, value_node in node.value:
            if key_node.value == "<<":
                if not seen_merge:
                    seen_merge = True
                    new_value.append((key_node, value_node))
                else:
                    warn(f"Duplicate merge key '<<' dropped in {node.start_mark}")
            else:
                new_value.append((key_node, value_node))
        node.value = new_value
        return super().flatten_mapping(node)


def warn(msg: str) -> None:
    print(f"Warning: {msg}")


def parse_frequency(freq_str: str) -> timedelta:
    freq_str = str(freq_str).strip()
    m = re.fullmatch(r"(\d+)([hHdDmM])?", freq_str)
    if not m:
        raise ValueError(f"Cannot parse frequency: {freq_str!r}")
    value = int(m.group(1))
    unit = (m.group(2) or "h").lower()
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "m":
        return timedelta(minutes=value)
    raise ValueError(f"Unknown frequency unit: {unit!r}")


def parse_date(date_str: str) -> datetime:
    date_str = str(date_str).strip()
    # Strip timezone suffix (+00:00, Z, etc.) - we ignore tz for date arithmetic.
    date_str_no_tz = re.sub(r"([+-]\d{2}:\d{2}|Z)$", "", date_str)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_str_no_tz, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str!r}")


def format_date(dt: datetime, original_str: str) -> str:
    original_str = str(original_str).strip()
    # Preserve any trailing timezone suffix from the original.
    tz_match = re.search(r"([+-]\d{2}:\d{2}|Z)$", original_str)
    tz_suffix = tz_match.group(1) if tz_match else ""
    if "T" in original_str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_suffix
    if " " in original_str:
        return dt.strftime("%Y-%m-%d %H:%M:%S") + tz_suffix
    return dt.strftime("%Y-%m-%d") + tz_suffix


def is_mars_like(d: dict) -> bool:
    return bool(MARS_KEYS & d.keys())


def update_name(data: dict, input_path: Path, suffix: str = "-test") -> None:
    """Update the top-level 'name' field by appending suffix, warn if it mismatches filename."""

    if "name" not in data:
        return
    original_name = str(data["name"])
    expected_name = input_path.stem
    if original_name != expected_name:
        warn(f"Recipe name {original_name!r} does not match filename {expected_name!r}")
    data["name"] = original_name + suffix


def _iter_dates(start: datetime, end: datetime, frequency: timedelta) -> list[datetime]:
    dates = []
    date = start
    while date <= end:
        dates.append(date)
        date += frequency
    return dates


def _iter_missing_entries(entries: Any, default_frequency: timedelta) -> Iterable[datetime]:
    if isinstance(entries, (list, tuple)):
        for entry in entries:
            yield from _iter_missing_entries(entry, default_frequency)
        return

    if isinstance(entries, dict):
        if {"start", "end"} <= entries.keys():
            frequency = parse_frequency(entries.get("frequency", default_frequency))
            start = parse_date(entries["start"])
            end = parse_date(entries["end"])
            yield from _iter_dates(start, end, frequency)
            return
        raise ValueError(f"Unsupported missing range format: {entries!r}")

    if isinstance(entries, str) and "/" in entries:
        start, end, step = entries.split("/")
        yield from _iter_dates(parse_date(start), parse_date(end), parse_frequency(step))
        return

    yield parse_date(entries)


def _existing_missing(node: dict, frequency: timedelta) -> set[datetime]:
    missing = set()
    for dt in _iter_missing_entries(node.get("missing", []), frequency):
        missing.add(dt)
    return missing


def _compress_missing_ranges(dates: set[datetime], frequency: timedelta) -> list[tuple[datetime, datetime]]:
    if not dates:
        return []

    ordered = sorted(dates)
    ranges = []
    start = ordered[0]
    end = ordered[0]

    for current in ordered[1:]:
        if current - end == frequency:
            end = current
        else:
            ranges.append((start, end))
            start = current
            end = current

    ranges.append((start, end))
    return ranges


def _updated_missing_ranges(node: dict, *, n_dates: int, include_last_dates: bool) -> list[dict[str, str]]:
    start = parse_date(node["start"])
    end = parse_date(node["end"])
    frequency = parse_frequency(node["frequency"])

    all_dates = _iter_dates(start, end, frequency)
    existing_missing = _existing_missing(node, frequency)
    available_dates = [d for d in all_dates if d not in existing_missing]

    retained = list(available_dates[:n_dates])
    if include_last_dates:
        retained.extend(available_dates[-n_dates:])
    retained = set(retained)

    missing_dates = {d for d in all_dates if d not in retained}
    ranges = _compress_missing_ranges(missing_dates, frequency)

    return [
        {
            "start": format_date(range_start, node["start"]),
            "end": format_date(range_end, node["end"]),
        }
        for range_start, range_end in ranges
    ]


def walk(
    node,
    *,
    reduce_dates: bool,
    reduce_grid: bool,
    reduce_levels: bool,
    include_last_dates: bool,
    n_dates: int,
    n_levels: int,
) -> None:
    if isinstance(node, dict):
        # Keep start/end/frequency unchanged and reduce dates via missing ranges.
        if reduce_dates and {"start", "end", "frequency"} <= node.keys():
            try:
                missing_ranges = _updated_missing_ranges(node, n_dates=n_dates, include_last_dates=include_last_dates)
                if missing_ranges:
                    node["missing"] = missing_ranges
                elif "missing" in node:
                    del node["missing"]
            except ValueError as exc:
                warn(f"Could not reduce dates block: {exc}")

        if is_mars_like(node):
            if reduce_grid and "grid" in node:
                node["grid"] = "20./20."
            if reduce_levels:
                for key in LEVEL_KEYS:
                    if key in node:
                        levels = node[key]
                        if isinstance(levels, list) and len(levels) > n_levels:
                            node[key] = levels[:n_levels]

        for value in node.values():
            walk(
                value,
                reduce_dates=reduce_dates,
                reduce_grid=reduce_grid,
                reduce_levels=reduce_levels,
                include_last_dates=include_last_dates,
                n_dates=n_dates,
                n_levels=n_levels,
            )

    elif isinstance(node, list):
        for item in node:
            walk(
                item,
                reduce_dates=reduce_dates,
                reduce_grid=reduce_grid,
                reduce_levels=reduce_levels,
                include_last_dates=include_last_dates,
                n_dates=n_dates,
                n_levels=n_levels,
            )


def create_test_recipe(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    dates: bool = True,
    last_dates: bool = True,
    grid: bool = True,
    level: bool = True,
    n_dates: int = 4,
    n_levels: int = 2,
) -> Path:
    input_path = Path(input_path)
    if not dates and last_dates:
        raise ValueError("--last-dates cannot be used with --no-dates.")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}-test{input_path.suffix}"
    else:
        output_path = Path(output_path)

    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.Constructor = LenientConstructor

    with input_path.open() as f:
        data = yaml.load(f)

    update_name(data, input_path)

    walk(
        data,
        reduce_dates=dates,
        reduce_grid=grid,
        reduce_levels=level,
        include_last_dates=last_dates,
        n_dates=n_dates,
        n_levels=n_levels,
    )

    with output_path.open("w") as f:
        yaml.dump(data, f)

    return output_path


class CreateTestRecipe(Command):
    """Create a reduced test recipe from a full anemoi-datasets recipe YAML file."""

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser."""

        command_parser.add_argument(
            "--dates",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reduce dates to the first N values (default: enabled)",
        )
        command_parser.add_argument(
            "--last-dates",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Also include the last N dates in each date block (default: enabled)",
        )
        command_parser.add_argument(
            "--grid",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Set grid to 20./20. in MARS blocks (default: enabled)",
        )
        command_parser.add_argument(
            "--level",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reduce levelist/level to first N levels (default: enabled)",
        )

        command_parser.add_argument(
            "--n-dates", type=int, default=4, metavar="N", help="Number of dates to keep (default: 4)"
        )
        command_parser.add_argument(
            "--n-levels", type=int, default=2, metavar="N", help="Number of levels to keep (default: 2)"
        )

        command_parser.add_argument("input", help="Input recipe YAML file")
        command_parser.add_argument("output", nargs="?", help="Output file (default: <input-stem>-test.yaml)")

    def check(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if not args.dates and args.last_dates:
            parser.error("--last-dates cannot be used with --no-dates.")

    def run(self, args: Any) -> None:
        create_test_recipe(
            args.input,
            args.output,
            dates=args.dates,
            last_dates=args.last_dates,
            grid=args.grid,
            level=args.level,
            n_dates=args.n_dates,
            n_levels=args.n_levels,
        )


command = CreateTestRecipe