# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from functools import cached_property
from typing import Any

import numpy as np
import zarr

from ..base.statistics import StatisticsTask
from .tasks import FieldTaskMixin

LOG = logging.getLogger(__name__)


class Statistics(StatisticsTask, FieldTaskMixin):
    """A class to compute statistics for a dataset."""

    def __init__(
        self,
        path: str,
        config: dict | None = None,
        use_threads: bool = False,
        statistics_temp_dir: str | None = None,
        progress: Any = None,
        **kwargs: Any,
    ):
        """Initialize a Statistics instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        use_threads : bool, optional
            Whether to use threads.
        statistics_temp_dir : Optional[str], optional
            The directory for temporary statistics.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path, config)
        self.use_threads = use_threads
        self.progress = progress
        self.statistics_temp_dir = statistics_temp_dir

    def _run(self) -> None:
        """Run the statistics computation."""
        start, end = (
            self.dataset.zarr_metadata["statistics_start_date"],
            self.dataset.zarr_metadata["statistics_end_date"],
        )
        start, end = np.datetime64(start), np.datetime64(end)
        dates = self.dataset.anemoi_dataset.dates

        assert type(dates[0]) is type(start), (type(dates[0]), type(start))

        dates = [d for d in dates if d >= start and d <= end]
        dates = [d for i, d in enumerate(dates) if i not in self.dataset.anemoi_dataset.missing]
        variables = self.dataset.anemoi_dataset.variables
        stats = self.tmp_statistics.get_aggregated(dates, variables, self.allow_nans)

        LOG.info(stats)

        if not all(self.registry.get_flags(sync=False)):
            raise Exception(f"❗Zarr {self.path} is not fully built, not writing statistics into dataset.")

        for k in [
            "mean",
            "stdev",
            "minimum",
            "maximum",
            "sums",
            "squares",
            "count",
            "has_nans",
        ]:
            self.dataset.add_dataset(name=k, array=stats[k], dimensions=("variable",))

        self.registry.add_to_history("compute_statistics_end")
        LOG.info(f"Wrote statistics in {self.path}")

    @cached_property
    def allow_nans(self) -> bool | list:
        """Check if NaNs are allowed."""

        z = zarr.open(self.path, mode="r")
        if "allow_nans" in z.attrs:
            return z.attrs["allow_nans"]

        if "variables_with_nans" in z.attrs:
            return z.attrs["variables_with_nans"]

        warnings.warn(f"Cannot find 'variables_with_nans' of 'allow_nans' in {self.path}.")
        return True
