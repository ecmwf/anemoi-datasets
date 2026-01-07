# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np

LOG = logging.getLogger(__name__)

STATISTICS = ("mean", "minimum", "maximum", "stdev")


def _identity(x):
    return x


class _SimplerCollectorBase:
    def __init__(self, column) -> None:
        self._sum = np.float64(0.0)
        self._count = np.int64(0)
        self._min = np.float64(np.inf)
        self._max = -np.float64(np.inf)
        self._sumsq = np.float64(0.0)
        self._column = column

    def update(self, data: any) -> None:
        valid_data = data[~np.isnan(data)]
        if valid_data.size == 0:
            return

        self._sum += np.sum(valid_data)
        self._count += valid_data.size
        self._min = min(self._min, np.min(valid_data).astype(np.float64))
        self._max = max(self._max, np.max(valid_data).astype(np.float64))
        self._sumsq += np.sum(valid_data**2)

    def statistics(self) -> dict[str, float]:
        if self._count == 0:
            LOG.warning(f"Column {self._column}: no statistics collected")
            return {_: np.nan for _ in STATISTICS}

        assert isinstance(self._sum, np.float64)
        assert isinstance(self._count, np.int64)
        assert isinstance(self._min, np.float64)
        assert isinstance(self._max, np.float64)
        assert isinstance(self._sumsq, np.float64)

        mean = self._sum / self._count
        stdev = np.sqrt((self._sumsq / self._count) - (mean**2))

        assert isinstance(stdev, np.float64)
        assert isinstance(mean, np.float64)

        return {
            "mean": mean,
            "minimum": self._min,
            "maximum": self._max,
            "stdev": stdev,
        }


class _CollectorBase:
    def __init__(self, column) -> None:
        self._count = np.int64(0)
        self._min = np.float64(np.inf)
        self._max = -np.float64(np.inf)
        self._column = column
        self._mean = np.float64(0.0)
        self._m2 = np.float64(0.0)

    def update(self, data: any) -> None:
        valid_data = data[~np.isnan(data)]
        if valid_data.size == 0:
            return

        # Vectorized Welford's algorithm
        valid_data = valid_data.astype(np.float64).ravel()
        n = len(valid_data)

        if n == 0:
            return

        old_mean = self._mean
        old_count = self._count
        new_count = old_count + n

        # Batch mean
        batch_mean = np.mean(valid_data)

        # Update combined mean
        self._mean = (old_count * old_mean + n * batch_mean) / new_count

        # Update M2 using parallel algorithm formula
        batch_m2 = np.sum((valid_data - batch_mean) ** 2)
        delta = batch_mean - old_mean
        self._m2 = self._m2 + batch_m2 + (old_count * n * delta * delta) / new_count

        self._count = new_count

        # Update min/max
        self._min = min(self._min, np.min(valid_data))
        self._max = max(self._max, np.max(valid_data))

    def statistics(self) -> dict[str, float]:
        if self._count == 0:
            LOG.warning(f"Column {self._column}: no statistics collected")
            return {_: np.nan for _ in STATISTICS}

        assert isinstance(self._mean, np.float64)
        assert isinstance(self._count, np.int64)
        assert isinstance(self._min, np.float64)
        assert isinstance(self._max, np.float64)
        assert isinstance(self._m2, np.float64)

        # Compute variance using Welford's M2
        variance = self._m2 / self._count if self._count > 0 else np.float64(0.0)
        if variance < 0:
            # this could happen due to numerical errors
            relative_error = abs(variance) / max(abs(self._m2), abs(self._mean**2), 1e-100)
            if relative_error > 1e-6:
                msg = f"Negative variance {variance} for column {self._column}, {relative_error=}"
                msg += f" m2={self._m2}, count={self._count}, mean={self._mean}, min={self._min}, max={self._max}"
                LOG.error(msg)
                raise ValueError(msg)
            variance = 0.0
        stdev = np.sqrt(variance)

        assert isinstance(stdev, np.float64)

        return {
            "mean": self._mean,
            "minimum": self._min,
            "maximum": self._max,
            "stdev": stdev,
        }


class _Collector(_CollectorBase):
    pass


class _TendencyCollector(_CollectorBase):
    def __init__(self, column, name: str, delta: int) -> None:
        super().__init__(column)
        self._name = name
        self._delta = delta
        self._queue = []

    def update(self, data):
        data = data.astype(np.float64)

        if len(self._queue) < self._delta:
            self._queue.append(data)
            return

        previous = self._queue.pop(0)
        super().update(data - previous)
        self._queue.append(data)


def _all(dates):
    return range(len(dates))


class StatisticsCollector:

    def __init__(
        self,
        variables_names: list[str] | None = None,
        allow_nans: bool = False,
        filter=_all,
        tendencies: list[int] | None = None,
    ) -> None:
        self._filter = filter

        self._collectors = None
        self._variables_names = variables_names
        self._allow_nans = allow_nans
        self._tendencies = tendencies or {}
        self._tendencies_collectors = {}

    def collect(self, array: any, dates: any) -> None:

        if self._collectors is None:
            names = self._variables_names
            self._collectors = [_Collector(str(_) if names is None else names[_]) for _ in range(array.shape[1])]

            for name, delta in self._tendencies.items():
                self._tendencies_collectors[name] = [
                    _TendencyCollector(str(_) if names is None else names[_], name, delta)
                    for _ in range(array.shape[1])
                ]

        for j in range(array.shape[1]):
            values = array[:, j]
            self._collectors[j].update(values)

            for c in self._tendencies_collectors.values():
                c[j].update(values)

    def statistics(self) -> list[dict[str, float]]:
        if self._collectors is None:
            LOG.warning("No statistics collected")
            return {_: np.array([np.nan]) for _ in STATISTICS}

        result = {key: [] for key in STATISTICS}
        for collector in self._collectors:
            stats = collector.statistics()
            for key in STATISTICS:
                result[key].append(stats[key])

        for name, collectors in self._tendencies_collectors.items():

            tendencies = {key: [] for key in STATISTICS}

            for collector in collectors:
                stats = collector.statistics()
                for key in STATISTICS:
                    tendencies[key].append(stats[key])

            for key in STATISTICS:
                result[f"statistics_tendencies_{name}_{key}"] = tendencies[key]

        for key in list(result.keys()):
            result[key] = np.array(result[key])

        return result
