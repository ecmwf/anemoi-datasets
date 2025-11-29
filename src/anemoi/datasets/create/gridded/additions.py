# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
import warnings
from functools import cached_property
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets import MissingDateError
from anemoi.datasets import open_dataset
from anemoi.datasets.create.chunks import ChunkFilter
from anemoi.datasets.create.persistent import build_storage
from anemoi.datasets.create.statistics import Summary
from anemoi.datasets.create.statistics import check_variance
from anemoi.datasets.create.statistics import compute_statistics
from anemoi.datasets.create.statistics import fix_variance

from ..base.finalise_additions import FinaliseAdditionsTask
from ..base.init_additions import InitAdditionsTask
from ..base.load_additions import LoadAdditionsTask
from .tasks import FieldTaskMixin

LOG = logging.getLogger(__name__)


class AdditionsMixin:
    """A mixin class to handle dataset additions."""

    def skip(self) -> bool:
        """Check if the additions should be skipped.

        Returns
        -------
        bool
            Whether to skip the additions.
        """
        frequency = frequency_to_timedelta(self.dataset.anemoi_dataset.frequency)
        if not self.delta.total_seconds() % frequency.total_seconds() == 0:
            LOG.debug(f"Delta {self.delta} is not a multiple of frequency {frequency}. Skipping.")
            return True

        if self.dataset.zarr_metadata.get("build", {}).get("additions", None) is False:
            LOG.warning(f"Additions are disabled for {self.path} in the recipe.")
            return True

        return False

    @cached_property
    def tmp_storage_path(self) -> str:
        """Get the path to the temporary storage."""
        name = "storage_for_additions"
        if self.delta:
            name += frequency_to_string(self.delta)
        return os.path.join(f"{self.path}.{name}.tmp")

    def read_from_dataset(self) -> None:
        """Read data from the dataset."""
        self.variables = self.dataset.anemoi_dataset.variables
        self.frequency = frequency_to_timedelta(self.dataset.anemoi_dataset.frequency)
        start = self.dataset.zarr_metadata["statistics_start_date"]
        end = self.dataset.zarr_metadata["statistics_end_date"]
        self.start = datetime.datetime.fromisoformat(start)
        self.end = datetime.datetime.fromisoformat(end)

        ds = open_dataset(self.path, start=self.start, end=self.end)
        self.dates = ds.dates
        self.total = len(self.dates)

        idelta = self.delta.total_seconds() // self.frequency.total_seconds()
        assert int(idelta) == idelta, idelta
        idelta = int(idelta)
        self.ds = DeltaDataset(ds, idelta)


class DeltaDataset:
    """A class to represent a dataset with delta values."""

    def __init__(self, ds: Any, idelta: int):
        """Initialize a DeltaDataset instance.

        Parameters
        ----------
        ds : Any
            The dataset.
        idelta : int
            The delta value.
        """
        self.ds = ds
        self.idelta = idelta

    def __getitem__(self, i: int) -> Any:
        """Get an item from the dataset.

        Parameters
        ----------
        i : int
            The index.

        Returns
        -------
        Any
            The item.
        """
        j = i - self.idelta
        if j < 0:
            raise MissingDateError(f"Missing date {j}")
        return self.ds[i : i + 1, ...] - self.ds[j : j + 1, ...]


class _InitAdditions(InitAdditionsTask, FieldTaskMixin, AdditionsMixin):
    """A class to initialize dataset additions."""

    def __init__(self, path: str, delta: str, use_threads: bool = False, progress: Any = None, **kwargs: Any):
        """Initialize an _InitAdditions instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        delta : str
            The delta value.
        use_threads : bool, optional
            Whether to use threads.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress

    def _run(self) -> None:
        """Run the additions initialization."""
        if self.skip():
            LOG.info(f"Skipping delta={self.delta}")
            return

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=True)
        self.tmp_storage.delete()
        self.tmp_storage.create()
        LOG.info(f"Dataset {self.tmp_storage_path} additions initialised.")

    def cleanup(self) -> None:
        """Clean up the temporary storage."""
        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        self.tmp_storage.delete()
        LOG.info(f"Cleaned temporary storage {self.tmp_storage_path}")


class _LoadAdditions(LoadAdditionsTask, FieldTaskMixin, AdditionsMixin):
    """A class to run dataset additions."""

    def __init__(
        self,
        path: str,
        delta: str,
        parts: str | None = None,
        use_threads: bool = False,
        progress: Any = None,
        **kwargs: Any,
    ):
        """Initialize a _LoadAdditions instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        delta : str
            The delta value.
        parts : Optional[str], optional
            The parts to load.
        use_threads : bool, optional
            Whether to use threads.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress
        self.parts = parts

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        LOG.info(f"Writing in {self.tmp_storage_path}")

    def _run(self) -> None:
        """Run the additions."""
        if self.skip():
            LOG.info(f"Skipping delta={self.delta}")
            return

        self.read_from_dataset()

        chunk_filter = ChunkFilter(parts=self.parts, total=self.total)
        for i in range(0, self.total):
            if not chunk_filter(i):
                continue
            date = self.dates[i]
            try:
                arr = self.ds[i]
                stats = compute_statistics(arr, self.variables, allow_nans=self.allow_nans)
                self.tmp_storage.add([date, i, stats], key=date)
            except MissingDateError:
                self.tmp_storage.add([date, i, "missing"], key=date)
        self.tmp_storage.flush()
        LOG.debug(f"Dataset {self.path} additions run.")

    def allow_nans(self) -> bool:
        """Check if NaNs are allowed.

        Returns
        -------
        bool
            Whether NaNs are allowed.
        """
        if self.dataset.anemoi_dataset.metadata.get("allow_nans", False):
            return True

        variables_with_nans = self.dataset.anemoi_dataset.metadata.get("variables_with_nans", None)
        if variables_with_nans is not None:
            return variables_with_nans
        warnings.warn(f"❗Cannot find 'variables_with_nans' in {self.path}, assuming nans allowed.")
        return True


class _FinaliseAdditions(FinaliseAdditionsTask, FieldTaskMixin, AdditionsMixin):
    """A class to finalize dataset additions."""

    def __init__(self, path: str, delta: str, use_threads: bool = False, progress: Any = None, **kwargs: Any):
        """Initialize a _FinaliseAdditions instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        delta : str
            The delta value.
        use_threads : bool, optional
            Whether to use threads.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        LOG.info(f"Reading from {self.tmp_storage_path}.")

    def _run(self) -> None:
        """Run the additions finalization."""
        if self.skip():
            LOG.info(f"Skipping delta={self.delta}.")
            return

        self.read_from_dataset()

        shape = (len(self.dates), len(self.variables))
        agg = dict(
            minimum=np.full(shape, np.nan, dtype=np.float64),
            maximum=np.full(shape, np.nan, dtype=np.float64),
            sums=np.full(shape, np.nan, dtype=np.float64),
            squares=np.full(shape, np.nan, dtype=np.float64),
            count=np.full(shape, -1, dtype=np.int64),
            has_nans=np.full(shape, False, dtype=np.bool_),
        )
        LOG.debug(f"Aggregating {self.__class__.__name__} statistics on shape={shape}. Variables : {self.variables}")

        found = set()
        ifound = set()
        missing = set()
        for _date, (date, i, stats) in self.tmp_storage.items():
            assert _date == date
            if stats == "missing":
                missing.add(date)
                continue

            assert date not in found, f"Duplicates found {date}"
            found.add(date)
            ifound.add(i)

            for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
                agg[k][i, ...] = stats[k]

        assert len(found) + len(missing) == len(self.dates), (
            len(found),
            len(missing),
            len(self.dates),
        )
        assert found.union(missing) == set(self.dates), (
            found,
            missing,
            set(self.dates),
        )

        if len(ifound) < 2:
            LOG.warning(f"Not enough data found in {self.path} to compute {self.__class__.__name__}. Skipped.")
            self.tmp_storage.delete()
            return

        mask = sorted(list(ifound))
        for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            agg[k] = agg[k][mask, ...]

        for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            assert agg[k].shape == agg["count"].shape, (
                agg[k].shape,
                agg["count"].shape,
            )

        minimum = np.nanmin(agg["minimum"], axis=0)
        maximum = np.nanmax(agg["maximum"], axis=0)
        sums = np.nansum(agg["sums"], axis=0)
        squares = np.nansum(agg["squares"], axis=0)
        count = np.nansum(agg["count"], axis=0)
        has_nans = np.any(agg["has_nans"], axis=0)

        assert sums.shape == count.shape
        assert sums.shape == squares.shape
        assert sums.shape == minimum.shape
        assert sums.shape == maximum.shape
        assert sums.shape == has_nans.shape

        mean = sums / count
        assert sums.shape == mean.shape

        x = squares / count - mean * mean
        # x[- 1e-15 < (x / (np.sqrt(squares / count) + np.abs(mean))) < 0] = 0
        # remove negative variance due to numerical errors
        for i, name in enumerate(self.variables):
            x[i] = fix_variance(x[i], name, agg["count"][i : i + 1], agg["sums"][i : i + 1], agg["squares"][i : i + 1])
        check_variance(x, self.variables, minimum, maximum, mean, count, sums, squares)

        stdev = np.sqrt(x)
        assert sums.shape == stdev.shape

        self.summary = Summary(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            count=count,
            sums=sums,
            squares=squares,
            stdev=stdev,
            variables_names=self.variables,
            has_nans=has_nans,
        )
        LOG.info(f"Dataset {self.path} additions finalised.")
        # self.check_statistics()
        self._write(self.summary)
        self.tmp_storage.delete()

    def _write(self, summary: Summary) -> None:
        """Write the summary to the dataset.

        Parameters
        ----------
        summary : Summary
            The summary to write.
        """
        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            name = f"statistics_tendencies_{frequency_to_string(self.delta)}_{k}"
            self.dataset.add_dataset(name=name, array=summary[k], dimensions=("variable",))
        self.registry.add_to_history(f"compute_statistics_{self.__class__.__name__.lower()}_end")
        LOG.debug(f"Wrote additions in {self.path}")


def multi_addition(cls: type) -> type:
    """Create a class to handle multiple additions.

    Parameters
    ----------
    cls : type
        The class to handle additions.

    Returns
    -------
    type
        The class to handle multiple additions.
    """

    class MultiAdditions:
        def __init__(self, *args, **kwargs: Any):
            self.tasks = []

            for k in kwargs.pop("delta", []):
                self.tasks.append(cls(*args, delta=k, **kwargs))

            if not self.tasks:
                LOG.warning("No delta found in kwargs, no additions will be computed.")

        def run(self) -> None:
            """Run the additions."""
            for actor in self.tasks:
                actor.run()

    return MultiAdditions


InitAdditions = multi_addition(_InitAdditions)
LoadAdditions = multi_addition(_LoadAdditions)
FinaliseAdditions = multi_addition(_FinaliseAdditions)
