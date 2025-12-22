# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging

import numpy as np

LOG = logging.getLogger(__name__)

STATISTICS = ("mean", "minimum", "maximum", "stdev")


class _Collector:
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
        self._min = min(self._min, np.min(valid_data))
        self._max = max(self._max, np.max(valid_data))
        self._sumsq += np.sum(valid_data**2)

    def statistics(self) -> dict[str, float]:
        if self._count == 0:
            LOG.warning(f"Column {self._column}: no statistics collected")
            return {_: np.nan for _ in STATISTICS}

        mean = self._sum / self._count
        stdev = (self._sumsq / self._count) - (mean**2)
        return {
            "mean": mean,
            "minimum": self._min,
            "maximum": self._max,
            "stdev": stdev,
        }


class StatisticsCollector:

    def __init__(self, cutoff_date: datetime.datetime | None = None, columns_names: list[str] | None = None) -> None:
        self.cutoff_date = cutoff_date

        self._collectors = None
        self._columns_names = columns_names

    def collect(self, offset: int, array: any, dates: any) -> None:
        if not self.is_active(offset, array, dates):
            return

        if self._collectors is None:
            names = self._columns_names
            self._collectors = [_Collector(str(_) if names is None else names[_]) for _ in range(array.shape[1])]

        for i in range(array.shape[1]):
            self._collectors[i].update(array[:, i])

    def is_active(self, offset: int, array: any, dates: any) -> bool:
        return self.cutoff_date is None or dates[0] <= self.cutoff_date

    def statistics(self) -> list[dict[str, float]]:
        if self._collectors is None:
            LOG.warning("No statistics collected")
            return {_: np.array([np.nan]) for _ in STATISTICS}

        result = {key: [] for key in STATISTICS}
        for collector in self._collectors:
            stats = collector.statistics()
            for key in STATISTICS:
                result[key].append(stats[key])

        for key in STATISTICS:
            result[key] = np.array(result[key])

        return result


class NoStatisticsCollector(StatisticsCollector):
    def is_active(self, offset: int, array: any, dates: any) -> None:
        return False
