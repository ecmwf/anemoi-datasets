import logging
import os
from typing import Any

import numpy as np
import zarr

from ..creator import Creator
from ..dataset import Dataset
from ..statistics import StatisticsCollector
from .context import TabularContext

LOG = logging.getLogger(__name__)


class TabularCreator(Creator):

    allow_nans = True

    ######################################################

    def task_init(self):
        super().task_init()

    ######################################################

    def context(self):
        return TabularContext()

    def load_result(self, result: Any, dataset: Dataset) -> None:
        np.save(
            os.path.join(
                self.work_dir,
                f"{result.start_date}-{result.end_date}.npy",
            ),
            result.to_numpy(),
        )

    def task_finalise(self):
        from .finalise import finalise_tabular_dataset

        # TODO: use info from metadata, not minimal_input
        collector = StatisticsCollector(columns_names=self.minimal_input.variables)
        store = zarr.open(self.path, mode="a")

        finalise_tabular_dataset(
            store=store,
            work_dir=self.work_dir,
            date_indexing=self.date_indexing,
            statistic_collector=collector,
            delete_files=True,
        )

        for name in ("mean", "minimum", "maximum", "stdev"):
            store.create_dataset(
                name,
                data=collector.statistics()[name],
                shape=collector.statistics()[name].shape,
                dtype=collector.statistics()[name].dtype,
                overwrite=True,
            )

    def task_statistics(self):
        pass

    def task_size(self) -> int:
        return 0

    @property
    def date_indexing(self) -> str:
        return self.recipe.date_indexing

    ######################################################
    @property
    def check_name(self) -> str:
        return False

    def shape_and_chunks(self, dates: Any) -> tuple[int, ...]:
        total_shape = (len(dates), len(self.minimal_input.variables))
        chunks = (min(100, total_shape[0]), total_shape[1])
        return total_shape, chunks
