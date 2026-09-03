# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import Any

import yaml

from anemoi.datasets.create.recipe import Recipe
from anemoi.datasets.create.check import LICENCE_POLICIES
from anemoi.datasets.create.check import collect_recipe_sources
from anemoi.datasets.create.naming import check_dataset_name
from anemoi.datasets.create.tabular.validate import validate_date_ranges
from anemoi.datasets.usage.store import open_zarr_store

from . import Command

LOG = logging.getLogger(__name__)

__all__ = ["Check"]


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
            "--recipe",
            help="Checks that recipe file is valid and that it is not missing required fields.",
        )

        exclusive_group.add_argument(
            "--zarr",
            help="Specify the Zarr archive to check.",
        )

        exclusive_group.add_argument(
            "--metadata",
            help="Specify the metadata file to check.",
        )

        exclusive_group.add_argument(
            "--index",
            help="Specify the index file to check.",
        )

        command_parser.add_argument(
            "--name",
            action="store_true",
            default=False,
            help="When used with --recipe, also check that the dataset name follows naming conventions.",
        )

        command_parser.add_argument(
            "--licence",
            action="store_true",
            default=False,
            help="When used with --recipe, also check that the licence is present and allowlisted.",
        )

    def run(self, args: Any) -> None:

        if args.recipe:
            self._check_recipe(args.recipe, check_name=args.name, check_licence=args.licence)

        if args.metadata:
            self._check_metadata(args.metadata)

        if args.zarr:
            self._check_zarr(args.zarr)

        if args.index:
            self._check_index(args.index, check_name=args.name, check_licence=args.licence)

    def _check_metadata(self, metadata: str) -> None:
        raise NotImplementedError("Metadata checking is not implemented yet.")

    def _check_recipe(self, recipe: str, *, check_name: bool = False, check_licence: bool = False) -> None:

        recipe_path = Path(recipe)
        recipe_filename = recipe_path.stem
        recipe_dict = yaml.safe_load(open(recipe, encoding="utf-8"))
        recipe_obj = Recipe.from_dict(recipe_dict)  # Validate recipe using pydantic model

        if recipe_filename != recipe_obj.name:
            raise ValueError(
                f"Recipe filename '{recipe_filename}' does not match the name in the recipe file '{recipe_obj.name}'."
            )

        if check_name:
            self._check_name(recipe_obj.name)

        if check_licence:
            self._check_licence(recipe_path)

        logging.info("Recipe check completed successfully!")

    def _check_name(self, name: str) -> None:

        fail = False
        for message in check_dataset_name(name):
            print("Dataset name warning: %s", message)
            fail = True
        if fail:
            raise ValueError("Dataset name does not follow naming conventions.")

    def _check_licence(self, recipe_path: Path) -> None:
        parsed = yaml.safe_load(recipe_path.open())
        active_sources = collect_recipe_sources(parsed, set(LICENCE_POLICIES))
        for source in active_sources:
            LICENCE_POLICIES[source].validate(parsed)

    def _check_zarr(self, zarr: str) -> None:
        raise NotImplementedError("Zarr archive checking is not implemented yet.")

    def _check_index(self, name: str, *, check_name: bool = False, check_licence: bool = False) -> None:
        if check_name or check_licence:
            raise NotImplementedError("--name and --licence are not supported with --index.")
        store = open_zarr_store(name)
        data = store["data"]
        index = store["date_index_ranges"]
        validate_date_ranges(data, index)

command = Check
