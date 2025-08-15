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
import sys
from typing import Any

import yaml

from ..dumper import yaml_dump
from . import Command

LOG = logging.getLogger(__name__)


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

        config = make_dates(config)

        text = yaml_dump(config, sort_keys=False, indent=2, width=120, order=ORDER)
        # with open(args.path + ".tmp", "w") as f:
        f = sys.stdout
        for i, line in enumerate(text.splitlines()):
            if i and line and line[0] not in (" ", "-"):
                line = "\n" + line
            print(line, file=f)

        # os.rename(args.path + ".tmp", args.path)


command = Recipe
