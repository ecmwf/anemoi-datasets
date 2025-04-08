# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from typing import Any

import yaml

from anemoi.datasets.create.check import DatasetName

from . import Command

LOG = logging.getLogger(__name__)


class Check(Command):
    """Check if a dataset name follow naming conventions."""

    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add command line arguments to the parser.

        Parameters
        ----------
        command_parser : Any
            The command line argument parser.
        """
        command_parser.add_argument(
            "--recipe",
            help="",
        )
        command_parser.add_argument(
            "--name",
            help="",
        )

    def run(self, args: Any) -> None:

        if args.recipe:
            recipe_filename = os.path.basename(args.recipe)
            recipe_name = os.path.splitext(recipe_filename)[0]
            in_recipe_name = yaml.safe_load(open(args.recipe, "r", encoding="utf-8"))["name"]
            if recipe_name != in_recipe_name:
                print(f"Recipe name {recipe_name} does not match the name in the recipe file {in_recipe_name}")

            name = recipe_name
            DatasetName(name=name).raise_if_not_valid()

        if args.name:
            name = args.name
            DatasetName(name=name).raise_if_not_valid()


command = Check
