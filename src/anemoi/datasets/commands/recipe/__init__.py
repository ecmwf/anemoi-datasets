# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import logging
import sys
from typing import Any

import yaml

from anemoi.datasets.create.fields import config_to_python
from anemoi.datasets.create.fields import validate_config

from .. import Command
from .format import format_recipe
from .migrate import migrate_recipe

LOG = logging.getLogger(__name__)


class Recipe(Command):
    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            Command parser object.
        """

        command_parser.add_argument("--validate", action="store_true", help="Validate recipe.")
        command_parser.add_argument("--format", action="store_true", help="Format the recipe.")
        command_parser.add_argument("--migrate", action="store_true", help="Migrate the recipe to the latest version.")
        command_parser.add_argument("--python", action="store_true", help="Convert the recipe to a Python script.")

        group = command_parser.add_mutually_exclusive_group()
        group.add_argument("--inplace", action="store_true", help="Overwrite the recipe file in place.")
        group.add_argument("--output", type=str, help="Output file path for the converted recipe.")

        command_parser.add_argument(
            "path",
            help="Path to recipe.",
        )

    def run(self, args: Any) -> None:

        if not args.validate and not args.format and not args.migrate and not args.python:
            args.validate = True

        with open(args.path) as file:
            config = yaml.safe_load(file)

        assert isinstance(config, dict)

        if args.validate:
            if args.inplace and (not args.format and not args.migrate and not args.python):
                argparse.ArgumentError(None, "--inplace is not supported with --validate.")

            if args.output and (not args.format and not args.migrate and not args.python):
                argparse.ArgumentError(None, "--output is not supported with --validate.")

            validate_config(config)
            LOG.info(f"{args.path}: Recipe is valid.")
            return

        if args.migrate:
            config = migrate_recipe(args, config)
            if config is None:
                LOG.info(f"{args.path}: No changes needed.")
                return

            args.format = True

        if args.format:
            formatted = format_recipe(args, config)
            assert "dates" in formatted
            f = sys.stdout
            if args.output:
                f = open(args.output, "w")

            if args.inplace:
                f = open(args.path, "w")

            print(formatted, file=f)
            f.close()

        if args.python:
            if args.inplace:
                argparse.ArgumentError(None, "Inplace conversion to Python is not supported.")

            if args.format:
                raise argparse.ArgumentError(None, "Formatting is not supported when converting to Python.")

            if args.output:
                with open(args.output, "w") as file:
                    file.write(config_to_python(config))
            else:
                print(config_to_python(config))


command = Recipe
