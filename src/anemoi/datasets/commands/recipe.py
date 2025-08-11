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
from .migrate import migrate

LOG = logging.getLogger(__name__)


class Recipe(Command):
    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            Command parser object.
        """

        command_parser.add_argument("--migrate", action="store_true", help="Migrate the recipe to the latest version.")

        command_parser.add_argument(
            "path",
            help="Path to recipe.",
        )

    def run(self, args: Any) -> None:
        from anemoi.datasets.create import config_to_python

        with open(args.path, "r") as file:
            config = yaml.safe_load(file)
            if args.migrate:
                config = migrate(config)

        print(config_to_python(config))


command = Recipe
