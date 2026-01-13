# (C) Copyright 2025 Anemoi contributors.
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
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import Any

import numpy as np
import tqdm
import zarr
from anemoi.utils.humanize import bytes_to_human
from anemoi.utils.sanitise import sanitise

from anemoi.datasets import open_dataset
from anemoi.datasets.create.dataset import Dataset
from anemoi.datasets.create.input.builder import InputBuilder
from anemoi.datasets.create.recipe import loader_recipe_from_yaml
from anemoi.datasets.create.recipe import loader_recipe_from_zarr
from anemoi.datasets.dates.groups import Groups

from .parts import PartFilter

LOG = logging.getLogger(__name__)

VERSION = "0.4"

LOG = logging.getLogger(__name__)


class Creator(ABC):
    """Abstract base class for dataset creation workflows.

    Provides methods for initialisation, loading, metadata management, statistics, and additions handling.
    """

    def __init__(self, path: str, recipe: dict, **kwargs: Any) -> None:
        """Initialise the Creator object.

        Parameters
        ----------
        path : str
            Path to the dataset.
        config : dict
            Main configuration dictionary.
        **kwargs : Any
            Additional keyword arguments for customisation.
        """
        # Catch all floating point errors, including overflow, sqrt(<0), etc

        np.seterr(all="raise", under="warn")

        self.path = path
        self.recipe = recipe

        self.parts = kwargs.pop("parts", None)

        self.kwargs = kwargs
        self.work_dir = kwargs.get("work_dir", self.path + ".work_dir")
        LOG.info(f"Using work dir: {os.path.abspath(self.work_dir)}")

    #####################################################

    @classmethod
    def from_recipe(cls, recipe: dict | str, **kwargs: Any) -> "Creator":
        """Instantiate a Creator subclass from a configuration.

        Parameters
        ----------
        recipe : dict or str
            Configuration dictionary or path to configuration file.
        **kwargs : Any
            Additional keyword arguments for subclass initialisation.

        Returns
        -------
        Creator
            An instance of a Creator subclass.
        """

        if recipe is None:
            # Look for recipe in the zarr
            if "path" not in kwargs:
                raise ValueError("Path must be provided in kwargs if recipe is None.")

            recipe = loader_recipe_from_zarr(kwargs["path"])

        if isinstance(recipe, str):
            recipe = loader_recipe_from_yaml(recipe)

        for k, v in recipe.build.env.items():
            os.environ[k] = str(v)

        format_type = recipe.output.format
        match format_type:

            case "gridded":
                from .gridded.creator import GriddedCreator

                return GriddedCreator(recipe=recipe, **kwargs)
            case "tabular":
                from .tabular.creator import TabularCreator

                return TabularCreator(recipe=recipe, **kwargs)
            case _:
                raise ValueError(f"Unknown format type: {format_type}")

    #####################################################
    # Initialisation
    #####################################################

    def task_init(self) -> Dataset:
        """Run the initialisation process for the dataset."""

        dataset = Dataset(self.path, overwrite=self.kwargs.get("overwrite", False), create=True)
        self._cleanup_temporary_directories()

        LOG.info("Initialising dataset creation.")
        LOG.info(f"Dataset path: {self.path}")
        LOG.info(f"Groups: {len(self.groups)}")

        metadata = {}
        self.fill_metadata(metadata)
        dataset.update_metadata(metadata)

        assert "uuid" in metadata, "super().collect_metadata() was not called or did not set 'uuid'"

        self.check_dataset_name(self.path)

        # Initialize the dataset
        self.initialise_dataset(dataset)
        # Initialize progress tracking
        dataset.initalise_done_flags(len(self.groups))

    @abstractmethod
    def check_dataset_name(self, path: str) -> None:
        pass

    @abstractmethod
    def initialise_dataset(self, dataset: Dataset) -> None:
        pass

    def fill_metadata(self, metadata: dict) -> None:
        metadata["version"] = VERSION
        metadata["uuid"] = str(uuid.uuid4())

        metadata["description"] = self.recipe.description
        metadata["licence"] = self.recipe.licence
        metadata["attribution"] = self.recipe.attribution

        # Store the recipe in the metadata so it can be retrieved later by later steps
        # This entry will be deleted when the dataset is finalised
        # We use model_dump_json to have a JSON string, because Zarr sorts attrs keys

        model_dump = self.recipe.model_dump_json()
        metadata["_recipe"] = model_dump

        # Store a sanitised (no path, no urls,...) version of the recipe for the catalogue
        # This one will be kept in the finalised dataset metadata

        model_dump = json.loads(model_dump)
        recipe = sanitise(model_dump)

        # Remove stuff added by prepml
        allow_keys = set(model_dump.keys())
        for k in recipe.keys():
            if k not in allow_keys:
                recipe.pop(k, None)

        metadata["recipe"] = recipe

        ##############
        metadata["dtype"] = self.recipe.output.dtype

        #####
        # Call subclass
        self.collect_metadata(metadata)

    @abstractmethod
    def collect_metadata(self, metadata: dict) -> None:
        pass

    ######################################################
    # Main loading loop
    ######################################################

    def task_load(self) -> None:
        """Load data into the dataset, processing each group as required."""
        dataset = Dataset(self.path, update=True)

        total = dataset.total_todo()
        filter = PartFilter(parts=self.parts, total=total)

        for i, group in enumerate(self.groups):

            if not filter(i):
                LOG.info(f" -> Skipping {i} total={total} (filtered out)")
                continue

            if dataset.is_done(i):
                LOG.info(f" -> Skipping {i} total={total} (already done)")
                continue

            result = self.input.select(self.context(), argument=group)

            # There are several groups. There is one result to load for each group.
            self.load_result(result, dataset)

            # Mark group as done
            dataset.mark_done(i)

        dataset.add_provenance(name="provenance_load")

    ########################
    # Finalisation
    ########################

    def task_finalise(self) -> None:
        LOG.info("Finalising dataset.")
        dataset = Dataset(self.path, update=True)
        self.finalise_dataset(dataset)

    @abstractmethod
    def finalise_dataset(self, dataset: Dataset) -> None:
        pass

    ######################################################
    # Misc tasks
    ######################################################

    def task_patch(self) -> None:
        pass

    def task_size(self) -> None:
        dataset = Dataset(self.path, update=True)
        size, count = 0, 0
        bar = tqdm.tqdm(iterable=os.walk(self.path), desc=f"Computing size of {os.path.basename(self.path)}")
        for dirpath, _, filenames in bar:
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                size += os.path.getsize(file_path)
                count += 1

        LOG.info(f"Total size: {bytes_to_human(size)}")
        LOG.info(f"Total number of files: {count:,}")

        dataset.update_metadata(total_size=size, total_number_of_files=count)

        return

        """Compute and update the size and constant fields metadata for the dataset."""
        """Run the size computation."""
        from anemoi.datasets.create.size import compute_directory_sizes

        self.dataset = self.open_writable_dataset(self.path)

        metadata = compute_directory_sizes(self.path)
        self.update_metadata(**metadata)

        # Look for constant fields
        ds = open_dataset(self.path)
        constants = ds.computed_constant_fields()

        variables_metadata = self.dataset.zarr_metadata.get("variables_metadata", {}).copy()
        for k in constants:
            if k in variables_metadata:
                variables_metadata[k]["constant_in_time"] = True

        self.update_metadata(constant_fields=constants, variables_metadata=variables_metadata)

    ########################
    # Cleanup
    # #######################

    def task_cleanup(self) -> None:
        """Clean up temporary statistics and registry, and remove additions if specified."""
        dataset = Dataset(self.path, update=True)
        dataset.remove_group("_build")
        self._cleanup_temporary_directories()

    def _cleanup_temporary_directories(self) -> None:
        """Clean up temporary directories used during dataset creation."""

        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
            LOG.info(f"Removed temporary directory: {self.work_dir}")

    ########################
    def task_verify(self) -> None:
        LOG.info("BACK: Verifying dataset.")
        return
        """Run verification on the dataset and log the results."""
        """Run the verification."""
        self.dataset = self.open_writable_dataset(self.path)
        LOG.info(f"Verifying dataset at {self.path}")
        LOG.info(str(self.dataset.anemoi_dataset))

    ########################
    def task_statistics(self) -> None:
        LOG.info("Running statistics computation.")
        dataset = Dataset(self.path, update=True)
        recompute_statistics = self.kwargs.get("recompute_statistics", False)

        if all(name in dataset.store for name in ("mean", "minimum", "maximum", "stdev")) and not recompute_statistics:
            LOG.info("Statistics already present, skipping computation.")
            return

        self.compute_and_store_statistics(dataset)

    @abstractmethod
    def compute_and_store_statistics(self, dataset: Dataset) -> None:
        pass

    ########################

    def task_init_additions(self) -> None:
        LOG.info("BACK: Initialising additions.")
        return

    def task_load_additions(self) -> None:
        LOG.info("BACK: Loading additions.")
        return

    def task_finalise_additions(self) -> None:
        LOG.info("BACK: Finalising additions.")
        return

    @cached_property
    def groups(self) -> Groups:
        """Return the date groups for the dataset."""
        return Groups(**self.recipe.dates)

    @cached_property
    def minimal_input(self) -> Any:
        """Return a minimal input selection for a single date."""
        one_date = self.groups.one_date()
        return self.input.select(self.context(), one_date)

    @cached_property
    def input(self) -> InputBuilder:
        """Return the input builder for the dataset."""

        return InputBuilder(
            self.recipe.input,
            data_sources=self.recipe.data_sources or {},
        )

    @cached_property
    def variables_names(self) -> list[str]:
        """Get the variable names."""
        z = zarr.open(self.path, mode="r")
        return z.attrs["variables"]
