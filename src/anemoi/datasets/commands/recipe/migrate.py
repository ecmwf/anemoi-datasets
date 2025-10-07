# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import sys
from collections.abc import Sequence
from typing import Any

from glom import assign
from glom import delete
from glom import glom

from anemoi.datasets.create import validate_config
from anemoi.datasets.dumper import yaml_dump

LOG = logging.getLogger(__name__)


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


def find_paths_in_substrees(path, obj, cur_path=None):
    if cur_path is None:
        cur_path = []
    matches = []
    try:
        glom(obj, path)  # just to check existence
        matches.append(cur_path + path.split("."))
    except Exception:
        pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            matches.extend(find_paths_in_substrees(path, v, cur_path + [k]))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            matches.extend(find_paths_in_substrees(path, v, cur_path + [str(i)]))
    return matches


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
    "include.mars": "data_sources.mars.mars",
    "ensemble_dimension": "build.ensemble_dimension",
    "flatten_grid": "build.flatten_grid",
}

DELETE = [
    "purpose",
    # "input.join.0.label",
    "status",
    "common",
    "config_format_version",
    "aliases",
    # "platform",
    "loops.0.loop_a.applies_to",
    "loop.0.loop_a.applies_to",
    "dataset_status",
    "alias",
    "resources",
    "input.dates.<<",
    "input.dates.join.0.label.name",
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


def _delete(config, path):
    x = glom(config, path, default=MARKER)
    if x is MARKER:
        return
    delete(config, path)


def _move(config, path, new_path, result):
    x = glom(config, path, default=MARKER)
    if x is MARKER:
        return
    delete(result, path)
    assign(result, new_path, x, missing=dict)


def _fix_input_0(config):
    if isinstance(config["input"], dict):
        return

    input = config["input"]
    new_input = []

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
                if inherit.startswith("$"):
                    inherit = inherit[1:]
                inherited = blocks[inherit].copy()
                inherited.update(values)
                values = inherited

            if first is None:
                first = source_name

            blocks[block_name] = values.copy()

            new_input.append({SOURCES.get(source_name, source_name): values.copy()})
        else:
            assert False, f"Block {block_name} does not have 'kwargs': {values}"

        blocks[block_name] = values.copy()

    config["input"] = dict(join=new_input)


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

    print("Found loops:", entries)

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
        print(f"Fixing {'.'.join(p)}")
        assign(result, ".".join(p[:-1] + ["template"]), "${input.join.0.mars}", missing=dict)
        delete(result, ".".join(p))

    paths = find_paths(config, target_key="date", target_value="$dates")
    for p in paths:
        delete(result, ".".join(p))


def _fix_join(result: dict, config: dict) -> None:
    print("Fixing join...")
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


def _fix_sources(config: dict, what) -> None:

    input = config["input"]
    if what not in input:
        return

    join = input[what]
    new_join = []
    for j in join:
        assert isinstance(j, dict)
        assert len(j) == 1, j

        key, values = list(j.items())[0]

        key = SOURCES.get(key, key)

        new_join.append(
            {
                key: values,
            }
        )

    config["input"][what] = new_join
    config["input"][what] = new_join.copy()


def _assign(config, path, value):
    print(f"Assign {path} {value}")
    assign(config, path, value)


def _fix_chevrons(result: dict, config: dict) -> None:
    print("Fixing chevrons...")
    paths = find_chevrons(config)
    for p in paths:
        a = glom(config, ".".join(p))
        b = glom(config, ".".join(p[:-1]))
        delete(result, ".".join(p))
        a.update(b)
        assign(result, ".".join(p[:-1]), a)


def _fix_some(config: dict) -> None:

    paths = find_paths_in_substrees("label.function", config)
    for p in paths:
        parent = glom(config, ".".join(p[:-2]))
        node = glom(config, ".".join(p[:-1]))
        assert node
        _assign(config, ".".join(p[:-2]), node)

    paths = find_paths_in_substrees("constants.source_or_dataset", config)
    for p in paths:
        node = glom(config, ".".join(p[:-1]))
        node["template"] = node.pop("source_or_dataset")
        if node["template"] == "$previous_data":
            node["template"] = "${input.join.0.mars}"
    paths = find_paths_in_substrees("constants.template", config)
    for p in paths:
        node = glom(config, ".".join(p[:-1]))
        if node["template"] == "$pl_data":
            node["template"] = "${input.join.0.mars}"
    for d in ("date", "dates", "time"):
        paths = find_paths_in_substrees(d, config)
        for p in paths:
            if len(p) > 1:
                node = glom(config, ".".join(p[:-1]))
                if isinstance(node, dict) and isinstance(node[d], str) and node[d].startswith("$"):
                    del node[d]

    paths = find_paths_in_substrees("source.<<", config)
    for p in paths:
        parent = glom(config, ".".join(p[:-2]))
        node = glom(config, ".".join(p[:-1]))
        node.update(node.pop("<<"))
        parent[node.pop("name")] = node
        assert len(parent) == 2
        del parent["source"]

    paths = find_paths_in_substrees("label.mars", config)
    for p in paths:
        parent = glom(config, ".".join(p[:-2]))
        node = glom(config, ".".join(p[:-1]))
        assert node
        assign(config, ".".join(p[:-2]), node)

    paths = find_paths_in_substrees("input.dates.join", config)
    for p in paths:
        node = glom(config, ".".join(p))
        config["input"]["join"] = node
        del config["input"]["dates"]

    paths = find_paths_in_substrees("source.name", config)
    for p in paths:
        parent = glom(config, ".".join(p[:-2]))
        node = glom(config, ".".join(p[:-1]))
        name = node.pop("name")
        assign(config, ".".join(p[:-2]), {name: node})

    paths = find_paths_in_substrees("function.name", config)
    for p in paths:
        parent = glom(config, ".".join(p[:-2]))
        node = glom(config, ".".join(p[:-1]))
        name = node.pop("name")
        assert node
        assign(config, ".".join(p[:-2]), {name: node})


def _migrate(config: dict, n) -> dict:

    result = config.copy()

    _fix_input_0(result)
    # _fix_loops(result, config)
    # _fix_input_1(result, config)
    # _fix_join(result, config)
    # _fix_chevrons(result, config)
    # _fix_other(result, config)

    for k, v in MIGRATE.items():
        _move(config, k, v, result)

    _fix_some(result)
    _fix_sources(result, "join")

    for k in DELETE:
        _delete(result, k)

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

    try:

        validate_config(config)
        assert config.get("input", {})
        assert config.get("dates", {})
        assert not has_key(config, "label")
        assert not has_key(config, "kwargs")
        assert not has_value(config, "$previous_data")
        assert not has_value(config, "$pl_data")
        assert not has_value(config, "$dates")
        assert not has_key(config, "inherit")
        assert not has_key(config, "source_or_dataset")
        assert not has_key(config, "<<")

        for n in SOURCES.keys():
            assert not has_key(config, n), f"Source {n} found in config. Please update to {SOURCES[n]}."

    except Exception as e:
        print("Validation failed:")
        print(e)
        print(yaml_dump(config))
        sys.exit(1)


def migrate_recipe(args: Any, config) -> None:

    print(f"Migrating {args.path}")

    migrated = migrate(config)

    check(migrated)
    if migrated == config:
        return None

    return migrated
