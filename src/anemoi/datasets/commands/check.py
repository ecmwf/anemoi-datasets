# (C) Copyright 2024-2026 Anemoi contributors.
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

from anemoi.datasets.create.naming import check_dataset_name
from anemoi.datasets.create.tabular.validate import validate_date_ranges
from anemoi.datasets.usage.store import open_zarr_store

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

        exclusive_group.add_argument(
            "--index",
            help="Specify the index file to check.",
        )

        command_parser.add_argument(
            "--layout",
            choices=["gridded", "trajectories", "tabular"],
            help="Dataset layout to validate the name against. "
            "When omitted, the name is accepted if any layout matches.",
        )

    def run(self, args: Any) -> None:

        if args.recipe:
            self._check_recipe(args.recipe)

        if args.metadata:
            self._check_metadata(args.metadata)

        if args.name:
            self._check_name(args.name, layout=args.layout)

        if args.zarr:
            self._check_zarr(args.zarr)

        if args.index:
            self._check_index(args.index)

    def _check_metadata(self, metadata: str) -> None:
        pass

    def _check_recipe(self, recipe: str) -> None:

        recipe_filename = os.path.basename(recipe)
        recipe_name = os.path.splitext(recipe_filename)[0]
        data = yaml.safe_load(open(recipe, encoding="utf-8"))
        in_recipe_name = data["name"]
        if recipe_name != in_recipe_name:
            print(f"Recipe name {recipe_name} does not match the name in the recipe file {in_recipe_name}")

        # Layout is authoritative — read it from the recipe rather than guessing
        # it from the name.  ``None`` falls back to the try-all behaviour.
        layout = data.get("output", {}).get("layout")
        self._check_name(in_recipe_name, layout=layout)

    def _check_name(self, name: str, layout: str | None = None) -> None:

        fail = False
        for message in check_dataset_name(name, layout=layout):
            print(f"Dataset name warning: {message}")
            fail = True
        if fail:
            raise ValueError("Dataset name does not follow naming conventions.")

    def _check_zarr(self, zarr: str) -> None:
        raise NotImplementedError("Zarr archive checking is not implemented yet.")

    def _check_index(self, name: str) -> None:
        store = open_zarr_store(name)
        data = store["data"]
        index = store["date_index_ranges"]
        validate_date_ranges(data, index)


command = Check
