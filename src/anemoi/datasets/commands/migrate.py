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
            old = yaml.safe_load(file)

        while True:
            new = self.migrate(old)
            if new == old:
                break
            old = new

        print(yaml.safe_dump(new, sort_keys=False, indent=2, width=120))

    def migrate(self, config: dict) -> dict:
        result = config.copy()
        # if 'loop' in config:
        #     result.pop('loop')
        #     # config['loop'] = config['loop'].replace('(', '[').replace(')', ']')

        if "statistics_end" in config.get("output", {}):
            result.setdefault("statistics", {})
            result["statistics"]["end"] = result["output"].pop("statistics_end")

        result = {k: v for k, v in sorted(result.items(), key=order)}

        return result


command = Recipe
