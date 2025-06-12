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

        exclusive_group = command_parser.add_mutually_exclusive_group(required=True)

        exclusive_group.add_argument(
            "--name",
            help="Check a dataset name.",
        )

        exclusive_group.add_argument(
            "--recipe",
            help="Specify the recipe file to check.",
        )

        exclusive_group.add_argument(
            "--zarr",
            help="Specify the Zarr archive to check.",
        )

        exclusive_group.add_argument(
            "--metadata",
            help="Specify the metadata file to check.",
        )

    def run(self, args: Any) -> None:

        if args.recipe:
            self._check_recipe(args.recipe)

        if args.metadata:
            self._check_metadata(args.metadata)

        if args.name:
            self._check_name(args.name)

        if args.zarr:
            self._check_zarr(args.zarr)

    def _check_metadata(self, metadata: str) -> None:
        pass

    def _check_recipe(self, recipe: str) -> None:

        recipe_filename = os.path.basename(recipe)
        recipe_name = os.path.splitext(recipe_filename)[0]
        in_recipe_name = yaml.safe_load(open(recipe, "r", encoding="utf-8"))["name"]
        if recipe_name != in_recipe_name:
            print(f"Recipe name {recipe_name} does not match the name in the recipe file {in_recipe_name}")

        name = in_recipe_name
        DatasetName(name=name).raise_if_not_valid()

    def _check_name(self, name: str) -> None:

        DatasetName(name=name).raise_if_not_valid()

    def _check_zarr(self, zarr: str) -> None:

        from anemoi.datasets.check import check_zarr

        check_zarr(zarr)

        # ds = xr.open_dataset(zarr)
        # print(ds)


command = Check
