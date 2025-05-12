# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import yaml

from . import Command

LOG = logging.getLogger(__name__)

ORDER = ("name", "description", "licence", "input", "output", "statistics", "build")
ORDER = {k: i for i, k in enumerate(ORDER)}


def order(x: str) -> int:

    if x[0] not in ORDER:
        ORDER[x[0]] = len(ORDER)

    return ORDER[x[0]]


MIGRATE = {
    "output.statistics_end": "statistics.end",
    "input.dates.<<": "dates",
    "input.dates.join": "input.join",
    "input.dates": None,
    "has_nans": "statistics.allow_nans",
    "loop.dates.group_by": "build.group_by",
    "loop.dates": "dates",
    "copyright": "attribution",
}

SOURCES = {
    "oper-accumulations": "accumulations",
    "era5-accumulations": "accumulations",
    "constants": "forcings",
    "ensemble-perturbations": "recentre",
}


def _move(config, path, new_path, result):
    path = path.split(".")
    if new_path is not None:
        new_path = new_path.split(".")

    for k in path[:-1]:
        if k not in config:
            return
        config = config[k]

    if path[-1] not in config:
        return

    value = config.pop(path[-1])

    if new_path is None:
        return

    for k in new_path[:-1]:
        if k not in result:
            result[k] = {}
        result = result[k]

    result[new_path[-1]] = value


def _migrate(config: dict, n) -> dict:
    result = config.copy()
    for k, v in MIGRATE.items():
        _move(config, k, v, result)

    if isinstance(result["input"], list):
        assert n == 0
        join = []
        prev = {}
        for n in result["input"]:
            assert isinstance(n, dict), (n, type(n))
            assert len(n) == 1, (n, type(n))
            name = list(n.keys())[0]
            prev[name] = n[name]["kwargs"]
            if "inherit" in n[name]:
                i = n[name]["inherit"]
                n[name]["kwargs"].update(prev[i])
                n[name].pop("inherit")

            data = n[name]["kwargs"]

            src = data.pop("name", "mars")

            join.append({SOURCES.get(src, src): data})

        result["input"] = dict(join=join)

    if "join" in result["input"] and n == 0:
        join = result["input"].pop("join")
        new_join = []

        for j in join:

            if "label" in j:
                if isinstance(j["label"], str):
                    j.pop("label")
                else:
                    if j["label"] is not None:
                        j = j["label"]
                    j.pop("name", None)

            if "source" in j:
                j = j["source"]

            src = j.pop("name", "mars")
            data = j
            if "<<" in data:
                data.update(data.pop("<<"))

            for k, v in list(data.items()):
                if k in ("date", "time"):
                    if isinstance(v, str) and v.startswith("$"):
                        del data[k]

            new_join.append({SOURCES.get(src, src): data})

        result["input"]["join"] = new_join

    if "join" in result["input"]:
        for j in result["input"]["join"]:
            k = list(j.keys())[0]
            j[k].pop("name", None)

            if "source_or_dataset" in j[k]:
                j[k].pop("source_or_dataset", None)
                j[k]["template"] = "${input.0.join.0.mars}"

    result = {k: v for k, v in sorted(result.items(), key=order) if v}

    result.pop("loop", None)

    return result


def migrate(old: dict) -> dict:
    # return _migrate(old)
    for i in range(10):
        new = _migrate(old, i)
        if new == old:
            # print(json.dumps(new, indent=2, default=str))
            return new
        old = new

    return new


class Recipe(Command):
    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            Command parser object.
        """
        command_parser.add_argument(
            "path",
            help="Path to recipe.",
        )

    def run(self, args: Any) -> None:
        with open(args.path, "r") as file:
            config = yaml.safe_load(file)

        print(yaml.safe_dump(migrate(config), sort_keys=False, indent=2, width=120))


command = Recipe
