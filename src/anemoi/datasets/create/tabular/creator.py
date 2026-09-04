# (C) Copyright 2025-2026 Anemoi contributors.
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

import numpy as np

from ..creator import Creator
from ..dataset import Dataset
from .context import TabularContext

LOG = logging.getLogger(__name__)


class TabularCreator(Creator):

    allow_nans = True

    ######################################################

    def check_dataset_name(self, path: str) -> None:
        """Check the dataset name for validity.

        Parameters
        ----------
        path : str
            The path to the dataset to be checked.
        """
        from pathlib import Path

        from ..naming import check_dataset_name

        name = Path(path).stem

        for message in check_dataset_name(
            name,
            resolution=None,
            start_date=self.groups.first_date(),
            end_date=self.groups.last_date(),
            layout="tabular",
        ):
            LOG.warning("Dataset name warning: %s", message)

    def collect_metadata(self, metadata: dict) -> None:
        super().collect_metadata(metadata)
        # See if that can be combined with `gridded`

        variables = self.minimal_input.variables
        LOG.info(f"Found {len(variables)} variables : {', '.join(variables)}.")
        metadata["variables"] = [v for v in variables if not v.startswith("__")]
        metadata["meta_variables"] = [v for v in variables if v.startswith("__")]

        metadata["dimensions"] = ["dates", "variables"]

        assert (
            variables == metadata["meta_variables"] + metadata["variables"]
        ), "Variables should be partitioned into variables and meta_variables without overlap, meta_variables must be first."

    def initialise_dataset(self, dataset: Dataset) -> None:
        """Initialise the dataset arrays and coordinates for tabular data.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to be initialised with arrays and coordinates.
        """
        pass

    ######################################################

    def context(self) -> TabularContext:
        return TabularContext(self.recipe)

    def load_result(self, result: Any, dataset: Dataset) -> None:
        """Load the result into the dataset by saving it as a NumPy file.

        Parameters
        ----------
        result : Any
            The result object containing the data to be loaded.
        dataset : Dataset
            The dataset object into which the result will be loaded.
        """
        os.makedirs(self.work_dir, exist_ok=True)

        # Guard against the pipeline producing columns that differ (in identity
        # or order) from the schema recorded at initialisation. The fragments
        # are saved as bare arrays, so this is the only place the column names
        # still exist; checking here also catches a task running mismatched code.
        expected = dataset.store.attrs["meta_variables"] + dataset.store.attrs["variables"]
        actual = result.variables
        assert actual == expected, (
            f"Column schema mismatch for {result.start_range}..{result.end_range}:\n"
            f"  pipeline produced: {actual}\n"
            f"  metadata declares: {expected}"
        )

        # Split large arrays into multiple files so that the finalisation step
        # does not need to load huge files into memory.

        # TODO: read value from recipe

        array = result.to_numpy()
        if array.shape[0] == 0:
            np.save(os.path.join(self.work_dir, f"{result.start_range}-{result.end_range}.npy"), array)
            return

        # Record the row density over time now, while the dates are in memory, so `finalise` can size
        # the zarr chunks without re-reading the fragments (see compute_rows_per_chunk).
        from .finalise import write_fragment_density

        write_fragment_density(self.work_dir, f"{result.start_range}-{result.end_range}", array)

        one_row_size = array.shape[1] * array.itemsize
        rows_per_file = max(round(self.recipe.build.max_fragment_size / one_row_size), 1)

        # Split on the change of date/time so we minimmise the numper of depluplications done in the "finalise" step

        # Find indices when the date or time changes
        mask = np.any(array[1:, :2] != array[:-1, :2], axis=1)
        change_indices = np.where(mask)[0] + 1

        def partition(start, end):
            # Base Case: Segment fits
            if (end - start) <= rows_per_file:
                return [slice(int(start), int(end))]

            # Find change points strictly within the current range (start, end)

            idx_start = np.searchsorted(change_indices, start, side="right")
            idx_end = np.searchsorted(change_indices, end, side="left")
            valid_changes = change_indices[idx_start:idx_end]

            if valid_changes.size > 0:
                # Find the change point closest to the midpoint for a balanced tree
                mid = (start + end) // 2
                split_idx = valid_changes[np.argmin(np.abs(valid_changes - mid))]
            else:
                # No change points exist in this range; force a split at max_rows
                split_idx = start + rows_per_file

            # Recurse
            return partition(start, split_idx) + partition(split_idx, end)

        partitions = partition(0, len(array))

        # for i, row_start in enumerate(range(0, array.shape[0], rows_per_file)):
        #     row_end = min(row_start + rows_per_file, array.shape[0])

        #     np.save(
        #         os.path.join(self.work_dir, f"{result.start_range}-{result.end_range}-{i:04d}.npy"),
        #         array[row_start:row_end],
        #     )

        for i, part in enumerate(partitions):
            LOG.info(
                f"{result.start_range}-{result.end_range}: Saving rows {part.start} to {part.stop} as part {i:04d} (len={part.stop - part.start}, max={rows_per_file})."
            )
            np.save(
                os.path.join(self.work_dir, f"{result.start_range}-{result.end_range}-{i:04d}.npy"),
                array[part],
            )

    def finalise_dataset(self, dataset: Dataset) -> None:
        """Finalise the dataset in a single process (backward-compatible full path).

        Runs the prepare, load (all fragments) and tidy stages in sequence. Splitting these
        stages across processes is done via ``finalise_prepare`` / ``finalise_load`` /
        ``finalise_tidy`` (see the ``anemoi-datasets finalise --prepare/--load/--tidy`` flags).

        Parameters
        ----------
        dataset : Dataset
            The dataset object to be finalised.
        """
        self.finalise_prepare(dataset)
        self.finalise_rows_per_chunk(dataset)
        self._finalise_load(dataset, parts=None)
        self.finalise_tidy(dataset)

    def finalise_prepare(self, dataset: Dataset) -> None:
        """Prepare stage: dedup, compute shape, create the zarr array and write the manifest."""
        from .finalise import prepare_tabular_dataset

        prepare_tabular_dataset(
            dataset=dataset,
            work_dir=self.work_dir,
            recipe=self.recipe,
            variables_names=self.variables_names,
            delete_files=self.recipe.build.delete_files,
            offset=4,
        )

    def finalise_rows_per_chunk(self, dataset: Dataset) -> None:
        """Choose (or just report) the optimal rows-per-chunk for tabular iteration windows.

        With ``--print`` it prints the optimum for every ``build.chunk_windows`` and changes nothing.
        Otherwise, when ``output.auto_rows_per_chunk`` is set and ``output.rows_per_chunk`` is still
        None, it computes the optimum for that single window, stores it as ``output.rows_per_chunk``
        and re-chunks the (still empty) data array. If ``rows_per_chunk`` is already set or
        ``auto_rows_per_chunk`` is null, it does nothing — unless invoked as the explicit
        ``finalise --rows-per-chunk`` stage, in which case it raises.
        """
        output = self.recipe.output

        if self.rows_per_chunk_print:
            print(self._compute_rows_per_chunk(dataset, self.recipe.build.chunk_windows))
            return

        explicit = self.finalise_stage == "rows_per_chunk"

        if output.rows_per_chunk is not None:
            message = f"output.rows_per_chunk is already set to {output.rows_per_chunk:,}."
            if explicit:
                raise ValueError(f"{message} Nothing to compute; use --print to just report the optima.")
            LOG.info(f"{message} Keeping the user's value; skipping automatic rows-per-chunk.")
            return

        if output.auto_rows_per_chunk is None:
            message = "output.auto_rows_per_chunk is null and output.rows_per_chunk is unset."
            if explicit:
                raise ValueError(f"{message} Set one of them, or use --print to just report the optima.")
            LOG.info(f"{message} Skipping automatic rows-per-chunk.")
            return

        window = output.auto_rows_per_chunk
        value = int(self._compute_rows_per_chunk(dataset, [window])[str(window)])

        # Persist the chosen value in the live recipe and both stored recipe copies (`recipe` is kept
        # in the finalised metadata, `_recipe` is used by later build steps before cleanup)...
        output.rows_per_chunk = value
        for key in ("_recipe", "recipe"):
            meta = dataset.get_metadata(key)
            if not meta:
                continue
            meta.setdefault("output", {})["rows_per_chunk"] = value
            dataset.update_metadata(**{key: meta})

        # ... and recreate the empty data array with the matching chunking before the load stage.
        from .finalise import set_rows_per_chunk

        set_rows_per_chunk(dataset, self.work_dir, value)
        LOG.info(f"Set output.rows_per_chunk={value:,} from the {window} window and re-chunked the data array.")

    def _compute_rows_per_chunk(self, dataset: Dataset, windows: Any) -> dict:
        from .finalise import compute_rows_per_chunk

        build = self.recipe.build
        return compute_rows_per_chunk(
            dataset=dataset,
            work_dir=self.work_dir,
            recipe=self.recipe,
            windows=windows,
            offset=build.chunk_alignment_offset,
            compression_ratio=build.chunk_compression_ratio,
            fs_read_min_bytes=build.fs_read_min_bytes,
            fs_read_max_bytes=build.fs_read_max_bytes,
            fs_latency_seconds=build.fs_read_latency_seconds,
            fs_bandwidth_bytes_per_s=build.fs_read_bandwidth_bytes_per_s,
        )

    def finalise_load(self, dataset: Dataset) -> None:
        """Load stage: write this process's ``--parts`` slice of fragments into the zarr array."""
        self._finalise_load(dataset, parts=self.parts)

    def _finalise_load(self, dataset: Dataset, parts: Any) -> None:
        from .finalise import load_tabular_dataset

        load_tabular_dataset(
            dataset=dataset,
            work_dir=self.work_dir,
            recipe=self.recipe,
            variables_names=self.variables_names,
            parts=parts,
            delete_files=self.recipe.build.delete_files,
        )

    def finalise_tidy(self, dataset: Dataset) -> None:
        """Tidy stage: merge partial statistics/date-ranges, build the index, set attrs and metadata."""
        from .finalise import tidy_tabular_dataset

        tidy_tabular_dataset(
            dataset=dataset,
            work_dir=self.work_dir,
            date_indexing=self.recipe.output.date_indexing,
            recipe=self.recipe,
            variables_names=self.variables_names,
            delete_files=self.recipe.build.delete_files,
        )

    def compute_and_store_statistics(self, dataset: Dataset) -> None:
        """Compute and store statistics for the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset object for which statistics will be computed and stored.
        """
        # TODO: implement if needed to recompute statistics
        raise NotImplementedError("Statistics are computed during finalisation for tabular datasets.")
