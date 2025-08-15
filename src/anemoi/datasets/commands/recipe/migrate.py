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
from collections.abc import Sequence
from typing import Any

import rich
import yaml
from glom import assign
from glom import delete
from glom import glom

from anemoi.datasets.create import validate_config

LOG = logging.getLogger(__name__)


class MyDumper(yaml.SafeDumper):
    pass


def find_paths(data, target_key=None, target_value=None, *path):

    matches = []

    if isinstance(data, dict):
        for k, v in data.items():
            if (target_key is not None and k == target_key) or (target_value is not None and v == target_value):
                matches.append(list(path) + [k])
            matches.extend(find_paths(v, target_key, target_value, *path, k))
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        for i, item in enumerate(data):
            matches.extend(find_paths(item, target_key, target_value, *path, str(i)))
    return matches


def find_chevrons(data, *path):

    matches = []

    if isinstance(data, dict):
        for k, v in data.items():
            if k == "<<":
                matches.append(list(path) + [k])
            matches.extend(find_chevrons(v, *path, k))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            matches.extend(find_chevrons(item, *path, str(i)))
    return matches


# Custom representer for datetime.date and datetime.datetime
def represent_date(dumper, data):
    if isinstance(data, datetime.date) and not isinstance(data, datetime.datetime):
        data = datetime.datetime(data.year, data.month, data.day, 0, 0, 0)
    # Ensure it's UTC
    if data.tzinfo is None:
        data = data.replace(tzinfo=datetime.timezone.utc)
    data = data.astimezone(datetime.timezone.utc)
    # Format as ISO 8601 with 'Z'
    iso_str = data.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", iso_str)


# Custom representer for multiline strings using the '|' block style
def represent_multiline_str(dumper, data):
    if "\n" in data:
        text_list = [line.rstrip() for line in data.splitlines()]
        fixed_data = "\n".join(text_list)
        return dumper.represent_scalar("tag:yaml.org,2002:str", fixed_data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# --- Represent short lists inline (flow style) ---
def represent_inline_list(dumper, data):
    # Flow style if list has <= 4 simple elements
    if (
        all(isinstance(i, (str, int, float, bool, type(None))) for i in data)
        and len(", ".join([str(x) for x in data])) + 2 <= 80
    ):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)


# Register custom representers
MyDumper.add_representer(datetime.date, represent_date)
MyDumper.add_representer(datetime.datetime, represent_date)
MyDumper.add_representer(str, represent_multiline_str)
MyDumper.add_representer(list, represent_inline_list)


def make_dates(config):
    if isinstance(config, dict):
        return {k: make_dates(v) for k, v in config.items()}
    if isinstance(config, list):
        return [make_dates(v) for v in config]
    if isinstance(config, str):
        try:
            return datetime.datetime.fromisoformat(config)
        except ValueError:
            return config
    return config


ORDER = (
    "name",
    "description",
    "dataset_status",
    "licence",
    "attribution",
    "env",
    "dates",
    "common",
    "data_sources",
    "input",
    "output",
    "statistics",
    "build",
    "platform",
)
ORDER = {k: i for i, k in enumerate(ORDER)}


def order(x: str) -> int:

    try:
        return ORDER[x[0]]
    except KeyError:
        rich.print(f"Unknown key: {x}")
        raise


MIGRATE = {
    "output.statistics_end": "statistics.end",
    "has_nans": "statistics.allow_nans",
    "loop.dates.group_by": "build.group_by",
    "loop.0.dates.group_by": "build.group_by",
    "loop.dates": "dates",
    "loop.0.dates": "dates",
    "copyright": "attribution",
    "dates.<<": "dates",
    "options.group_by": "build.group_by",
    "loops.0.loop_a.dates": "dates",
    "loop.0.loop_a.dates": "dates",
    "dates.stop": "dates.end",
    "dates.group_by": "build.group_by",
    "include": "data_sources",
    "ensemble_dimension": "build.ensemble_dimension",
    "flatten_grid": "build.flatten_grid",
}

DELETE = [
    "purpose",
    "input.join.0.label",
    "status",
    "common",
    "config_format_version",
    "aliases",
    "platform",
    "loops.0.loop_a.applies_to",
    "loop.0.loop_a.applies_to",
    "dataset_status",
    "alias",
    "resources",
]


SOURCES = {
    "oper-accumulations": "accumulations",
    "era5-accumulations": "accumulations",
    "ensemble-perturbations": "recentre",
    "ensemble_perturbations": "recentre",
    "perturbations": "recentre",
    "custom-regrid": "regrid",
}

MARKER = object()


def _delete(config, path, result):
    x = glom(config, path, default=MARKER)
    if x is MARKER:
        return
    rich.print(f"Deleting {path}={x}")
    delete(result, path)


def _move(config, path, new_path, result):
    x = glom(config, path, default=MARKER)
    if x is MARKER:
        return
    rich.print(f"Moving {path}={x} to {new_path}={x}")
    delete(result, path)
    assign(result, new_path, x, missing=dict)


def _fix_input_0(result, config):
    if isinstance(config["input"], dict):
        return

    input = config["input"]
    new_input = result["input"] = []

    blocks = {}
    first = None
    for block in input:
        assert isinstance(block, dict), block

        assert len(block) == 1, block

        block_name, values = list(block.items())[0]

        if "kwargs" in values:
            inherit = values.pop("inherit", None)
            assert len(values) == 1, values
            values = values["kwargs"]
            values.pop("date", None)
            source_name = values.pop("name", None)

            if inherit is not None:
                inherited = blocks[inherit].copy()
                inherited.update(values)
                values = inherited

            if "source_or_dataset" in values:
                values.pop("source_or_dataset", None)
                values["template"] = "${input.join.0." + first + "}"

            if first is None:
                first = source_name

            blocks[block_name] = values.copy()

            new_input.append({block_name: {SOURCES.get(source_name, source_name): values.copy()}})
        else:
            assert False, f"Block {block_name} does not have 'kwargs': {values}"

        blocks[block_name] = values.copy()

    config["input"] = result["input"].copy()


def _fix_input_1(result, config):
    if isinstance(config["input"], dict):
        return

    input = config["input"]
    join = []
    for k in input:
        assert isinstance(k, dict)
        assert len(k) == 1, f"Input key {k} is not a string: {input}"
        name, values = list(k.items())[0]
        join.append(values)

    result["input"] = {"join": join}
    config["input"] = result["input"].copy()


def remove_empties(config: dict) -> None:
    """Remove empty dictionaries and lists from the config."""
    if isinstance(config, dict):
        keys_to_delete = [k for k, v in config.items() if v in (None, {}, [], [{}])]

        for k in keys_to_delete:
            del config[k]

        for k, v in config.items():
            remove_empties(v)

    if isinstance(config, list):
        for item in config:
            remove_empties(item)


def _fix_loops(result: dict, config: dict) -> None:
    if "loops" not in config:
        return

    input = config["input"]
    loops = config["loops"]

    assert isinstance(loops, list), loops
    assert isinstance(input, list), input

    entries = {}
    dates_block = None
    for loop in loops:
        assert isinstance(loop, dict), loop
        assert len(loop) == 1, loop
        loop = list(loop.values())[0]
        applies_to = loop["applies_to"]
        dates = loop["dates"]
        assert isinstance(applies_to, list), (applies_to, loop)
        for a in applies_to:
            entries[a] = dates.copy()

        if "start" in dates:
            start = dates["start"]
        else:
            start = max(dates["values"])

        if "end" in dates or "stop" in dates:
            end = dates.get("end", dates.get("stop"))
        else:
            end = min(dates["values"])

        if dates_block is None:
            dates_block = {
                "start": start,
                "end": end,
            }

        if "frequency" in dates:
            if "frequency" not in dates_block:
                dates_block["frequency"] = dates["frequency"]
            else:
                assert dates_block["frequency"] == dates["frequency"], (dates_block["frequency"], dates["frequency"])

        dates_block["start"] = min(dates_block["start"], start)
        dates_block["end"] = max(dates_block["end"], end)

    concat = []
    result["input"] = {"concat": concat}

    rich.print("Found loops:", entries)

    for block in input:
        assert isinstance(block, dict), block
        assert len(block) == 1, block
        name, values = list(block.items())[0]
        assert name in entries, f"Loop {name} not found in loops: {list(entries.keys())}"
        dates = entries[name].copy()

        assert "kwargs" not in values

        concat.append(dict(dates=dates, **values))

    d = concat[0]["dates"]
    if all(c["dates"] == d for c in concat):
        join = []
        for c in concat:
            del c["dates"]
            join.append(c)
        result["input"] = {"join": join}

    del config["loops"]
    config["input"] = result["input"].copy()
    config["dates"] = dates_block.copy()
    del result["loops"]
    result["dates"] = dates_block


def _fix_other(result: dict, config: dict) -> None:
    paths = find_paths(config, target_key="source_or_dataset", target_value="$previous_data")
    for p in paths:
        rich.print(f"Fixing {'.'.join(p)}")
        assign(result, ".".join(p[:-1] + ["template"]), "${input.join.0.mars}", missing=dict)
        delete(result, ".".join(p))

    paths = find_paths(config, target_key="date", target_value="$dates")
    for p in paths:
        delete(result, ".".join(p))


def _fix_join(result: dict, config: dict) -> None:
    rich.print("Fixing join...")
    input = config["input"]
    if "dates" in input and "join" in input["dates"]:
        result["input"]["join"] = input["dates"]["join"]
        config["input"]["join"] = input["dates"]["join"].copy()

    if "join" not in input:
        return

    join = input["join"]
    new_join = []
    for j in join:
        assert isinstance(j, dict)
        assert len(j) == 1

        key, values = list(j.items())[0]

        if key not in ("label", "source"):
            return

        assert isinstance(values, dict), f"Join values for {key} should be a dict: {values}"
        if key == "label":
            j = values
            j.pop("name")
            key, values = list(j.items())[0]

        print(values)
        source_name = values.pop("name", "mars")
        new_join.append(
            {
                SOURCES.get(source_name, source_name): values,
            }
        )

    result["input"] = {"join": new_join}
    config["input"] = result["input"].copy()


def _fix_sources(result: dict, config: dict, what) -> None:

    input = config["input"]
    if what not in input:
        return

    join = input[what]
    new_join = []
    for j in join:
        assert isinstance(j, dict)
        assert len(j) == 1

        key, values = list(j.items())[0]

        key = SOURCES.get(key, key)

        new_join.append(
            {
                key: values,
            }
        )

    result["input"][what] = new_join
    config["input"][what] = new_join.copy()


def _fix_chevrons(result: dict, config: dict) -> None:
    rich.print("Fixing chevrons...")
    paths = find_chevrons(config)
    for p in paths:
        a = glom(config, ".".join(p))
        b = glom(config, ".".join(p[:-1]))
        delete(result, ".".join(p))
        a.update(b)
        assign(result, ".".join(p[:-1]), a)


def _migrate(config: dict, n) -> dict:

    result = config.copy()

    _fix_input_0(result, config)
    _fix_loops(result, config)
    _fix_input_1(result, config)
    _fix_join(result, config)
    _fix_sources(result, config, "join")
    _fix_chevrons(result, config)
    _fix_other(result, config)

    for k, v in MIGRATE.items():
        _move(config, k, v, result)

    for k in DELETE:
        _delete(config, k, result)

    remove_empties(result)

    return result


def migrate(old: dict) -> dict:

    for i in range(10):
        new = _migrate(old, i)
        if new == old:
            return new
        old = new

    return new


def has_key(config, key: str) -> bool:
    if isinstance(config, dict):
        if key in config:
            return True
        for k, v in config.items():
            if has_key(v, key):
                return True
    if isinstance(config, list):
        for item in config:
            if has_key(item, key):
                return True
    return False


def has_value(config, value: str) -> bool:
    if isinstance(config, dict):
        for k, v in config.items():
            if v == value:
                return True
            if has_value(v, value):
                return True

    if isinstance(config, list):
        for item in config:
            if item == value:
                return True
            if has_value(item, value):
                return True
    return config == value


def check(config):
    from anemoi.datasets.create import validate_config

    try:

        validate_config(config)
        assert config.get("input", {})
        assert config.get("dates", {})
        assert not has_key(config, "label")
        assert not has_key(config, "kwargs")
        assert not has_value(config, "$previous_data")
        assert not has_value(config, "$dates")
        assert not has_key(config, "inherit")
        assert not has_key(config, "source_or_dataset")
        assert not has_key(config, "<<")

        for n in SOURCES.keys():
            assert not has_key(config, n), f"Source {n} found in config. Please update to {SOURCES[n]}."

    except Exception as e:
        rich.print(f"Validation failed: {e}")
        rich.print(f"Config: {config}")
        raise


def migrate_recipe(args: Any, config) -> None:

    rich.print(f"Migrating {args.path}")

    try:
        validate_config(config)
        LOG.info(f"{args.path}: Validation successful.")
    except Exception:
        pass

    migrated = migrate(config)

    migrated = {k: v for k, v in sorted(migrated.items(), key=order) if v}

    check(migrated)
    if migrated == config:
        return None

    return migrated
