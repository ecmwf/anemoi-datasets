# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Standalone, single-process statistics for the ``compute`` command.

This module deliberately re-implements the numerically-stable (Welford / parallel)
statistics algorithm rather than importing it from :mod:`anemoi.datasets.create`.
It performs a simple chunked loop over a dataset, with no parallelism and no
on-disk caching, always accumulating in ``float64`` to preserve precision.
"""

import logging
from typing import Any
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)

#: Names of the statistics produced by this module.
STATISTICS = ("mean", "minimum", "maximum", "stdev")

#: Default number of time steps read per chunk.
DEFAULT_CHUNK_SIZE = 1


class Accumulator:
    """Numerically-stable accumulator for per-variable statistics.

    The accumulator follows the convention used throughout anemoi-datasets: axis 1
    of the incoming array is the *variable* axis; every other axis is treated as
    samples. NaN values are ignored on a per-variable basis. All internal state is
    kept in ``float64``.

    Parameters
    ----------
    variables : list of str
        Names of the variables (length must match axis 1 of the data).
    allow_nans : bool, optional
        If ``True``, NaNs are ignored per-variable. If ``False`` (default), a NaN
        in the data raises ``ValueError``.
    """

    def __init__(self, variables: list[str], allow_nans: bool = False) -> None:
        n = len(variables)
        self.variables = list(variables)
        self.allow_nans = bool(allow_nans)
        self._count = np.zeros(n, dtype=np.int64)
        self._mean = np.zeros(n, dtype=np.float64)
        self._m2 = np.zeros(n, dtype=np.float64)
        self._min = np.full(n, np.inf, dtype=np.float64)
        self._max = np.full(n, -np.inf, dtype=np.float64)

    def update(self, data: NDArray[Any]) -> None:
        """Accumulate a batch of data.

        Parameters
        ----------
        data : ndarray
            Array whose axis 1 is the variable axis. Remaining axes are samples.
        """
        if len(data) == 0:
            return

        data = np.asarray(data, dtype=np.float64)
        assert data.shape[1] == len(self.variables), (
            f"Array variable axis {data.shape[1]} does not match "
            f"variables count {len(self.variables)}"
        )

        if not self.allow_nans and np.isnan(data).any():
            bad = [self.variables[c] for c in range(data.shape[1]) if np.isnan(np.moveaxis(data, 1, 0)[c]).any()]
            raise ValueError(
                f"NaN values found for variable(s) {bad}; enable allow_nans to ignore them."
            )

        # Move the variable axis to the front, then flatten the sample axes.
        moved = np.moveaxis(data, 1, 0).reshape(len(self.variables), -1)

        for col_idx in range(moved.shape[0]):
            col = moved[col_idx]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue

            n = col.size
            old_count = self._count[col_idx]
            old_mean = self._mean[col_idx]
            new_count = old_count + n

            batch_mean = np.mean(col)
            batch_m2 = np.sum((col - batch_mean) ** 2)
            delta = batch_mean - old_mean

            self._mean[col_idx] = (old_count * old_mean + n * batch_mean) / new_count
            self._m2[col_idx] = self._m2[col_idx] + batch_m2 + (old_count * n * delta * delta) / new_count
            self._count[col_idx] = new_count
            self._min[col_idx] = min(self._min[col_idx], np.min(col))
            self._max[col_idx] = max(self._max[col_idx], np.max(col))

    def merge(self, other: "Accumulator") -> "Accumulator":
        """Merge another accumulator into a new one (parallel Welford).

        Parameters
        ----------
        other : Accumulator
            The accumulator to merge with; must share the same variables.

        Returns
        -------
        Accumulator
            A new accumulator holding the combined statistics.
        """
        if self.variables != other.variables:
            raise ValueError("Cannot merge accumulators with different variables")

        result = Accumulator(self.variables, allow_nans=self.allow_nans or other.allow_nans)
        count = self._count + other._count
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(count > 0, (self._count * self._mean + other._count * other._mean) / count, 0.0)
            delta = other._mean - self._mean
            m2 = np.where(
                count > 0,
                self._m2 + other._m2 + delta**2 * self._count * other._count / count,
                0.0,
            )
        result._count = count
        result._mean = mean
        result._m2 = m2
        result._min = np.minimum(self._min, other._min)
        result._max = np.maximum(self._max, other._max)
        return result

    def statistics(self) -> dict[str, NDArray[np.float64]]:
        """Return the final statistics.

        Returns
        -------
        dict
            Mapping with keys ``mean``, ``minimum``, ``maximum`` and ``stdev``,
            each an array indexed like :attr:`variables`. Variables with no data
            yield ``NaN``.
        """
        for col_idx, name in enumerate(self.variables):
            if self._count[col_idx] == 0:
                LOG.warning("Variable %s: no statistics collected", name)

        variance = np.divide(
            self._m2,
            self._count,
            out=np.full_like(self._m2, np.nan),
            where=self._count > 0,
        )

        # Guard against tiny negative variances from floating-point error.
        negative = variance < 0
        if np.any(negative):
            for col_idx in np.where(negative)[0]:
                var = variance[col_idx]
                scale = max(abs(self._m2[col_idx]), abs(self._mean[col_idx] ** 2), 1e-100)
                if abs(var) / scale > 1e-6:
                    raise ValueError(
                        f"Negative variance {var} for variable {self.variables[col_idx]} "
                        f"(m2={self._m2[col_idx]}, count={self._count[col_idx]})"
                    )
                variance[col_idx] = 0.0

        stdev = np.sqrt(variance)
        no_data = self._count == 0
        return {
            "mean": np.where(no_data, np.nan, self._mean),
            "minimum": np.where(no_data, np.nan, self._min),
            "maximum": np.where(no_data, np.nan, self._max),
            "stdev": np.where(no_data, np.nan, stdev),
        }


def iter_chunks(length: int, start: int, end: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield ``(lo, hi)`` index pairs covering ``[start, end)`` in steps of ``chunk_size``.

    Parameters
    ----------
    length : int
        Total length of the dataset (used to default/clamp ``end``).
    start : int
        First index to process.
    end : int or None
        One past the last index to process; ``None`` means ``length``.
    chunk_size : int
        Number of time steps per chunk.
    """
    if end is None:
        end = length
    end = min(end, length)
    chunk_size = max(1, int(chunk_size))
    for lo in range(start, end, chunk_size):
        yield lo, min(lo + chunk_size, end)


def compute_statistics(
    dataset: Any,
    start: int = 0,
    end: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    allow_nans: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Compute statistics over a dataset with a simple chunked loop.

    Parameters
    ----------
    dataset : Dataset
        An opened anemoi dataset supporting ``ds[lo:hi]``, ``len(ds)`` and
        ``ds.variables``.
    start : int, optional
        First time index to include.
    end : int, optional
        One past the last time index to include (defaults to ``len(dataset)``).
    chunk_size : int, optional
        Number of time steps read per chunk.
    allow_nans : bool, optional
        Whether to ignore NaNs per-variable.

    Returns
    -------
    dict
        The statistics as returned by :meth:`Accumulator.statistics`.
    """
    accumulator = Accumulator(list(dataset.variables), allow_nans=allow_nans)
    for lo, hi in _progress(iter_chunks(len(dataset), start, end, chunk_size)):
        accumulator.update(dataset[lo:hi])
    return accumulator.statistics()


def _progress(iterable: Iterator[Any]) -> Iterator[Any]:
    """Wrap ``iterable`` with a progress bar when :mod:`tqdm` is available."""
    try:
        import tqdm

        return tqdm.tqdm(list(iterable))
    except Exception:  # pragma: no cover - tqdm always present in practice
        return iterable
