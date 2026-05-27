# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os
import shutil
import uuid
from typing import Any

from anemoi.datasets.create.dataset import Dataset

from . import Command

LOG = logging.getLogger(__name__)


class Rename(Command):
    """Rename a dataset.

    Assign a new UUID, update the dataset name stored in the recipe metadata,
    and move the store to the new path.
    """

    def add_arguments(self, parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        parser : Any
            The command parser to which arguments are added.
        """
        parser.add_argument("source", help="Path of the dataset to rename (must end in .zarr).")
        parser.add_argument("target", help="New path for the dataset (must end in .zarr).")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite the target if it already exists.")

    def run(self, args: Any) -> None:
        """Execute the rename command.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """
        source = args.source.rstrip("/")
        target = args.target.rstrip("/")

        # --- validation ---
        if not source.endswith(".zarr") or not target.endswith(".zarr"):
            raise ValueError("Both source and target must end in '.zarr'.")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source dataset '{source}' does not exist.")
        if os.path.exists(target):
            if not args.overwrite:
                raise FileExistsError(f"Target '{target}' already exists. Use --overwrite.")
            LOG.warning(f"Overwriting existing dataset at '{target}'.")
            shutil.rmtree(target)

        new_name = os.path.splitext(os.path.basename(target))[0]

        # --- 1 & 2: update metadata in place on the source store ---
        dataset = Dataset(source, update=True)

        new_uuid = str(uuid.uuid4())
        old_uuid = dataset.get_metadata("uuid")

        # Update the name inside the recipe(s).
        for key in ("recipe", "_recipe"):
            recipe = dataset.get_metadata(key)
            if recipe is None:
                continue
            recipe = self._rename_in_recipe(recipe, new_name)
            dataset.update_metadata({key: recipe})

        # Track the chain of previous UUIDs (for provenance only).
        previous_uuids = dataset.get_metadata("previous_uuids", [])
        if old_uuid is not None:
            previous_uuids = previous_uuids + [old_uuid]

        dataset.update_metadata(uuid=new_uuid, previous_uuids=previous_uuids)

        LOG.info(f"Renamed '{source}' -> '{target}': uuid {old_uuid} -> {new_uuid}, recipe name -> '{new_name}'")

        # --- 3: move the store ---
        # Drop the Dataset reference so no file handles/locks are held during the move.
        del dataset
        shutil.move(source, target)

        # Best-effort cleanup of the stale source lock file.
        source_lock = source + ".lock"
        if os.path.exists(source_lock):
            try:
                os.remove(source_lock)
            except OSError as e:
                LOG.warning(f"Could not remove stale lock file '{source_lock}': {e}")

    @staticmethod
    def _rename_in_recipe(recipe: Any, new_name: str) -> Any:
        """Set the top-level ``name`` field of the recipe to *new_name*.

        Parameters
        ----------
        recipe : Any
            The recipe metadata, either a dict (the sanitised ``recipe`` attr)
            or a JSON string (the ``_recipe`` attr).
        new_name : str
            The new dataset name to store in the recipe.

        Returns
        -------
        Any
            The recipe with its ``name`` field updated, in the same form
            (dict or JSON string) as the input.
        """
        is_str = isinstance(recipe, str)
        data = json.loads(recipe) if is_str else dict(recipe)
        if "name" in data:
            data["name"] = new_name
        return json.dumps(data) if is_str else data


command = Rename
