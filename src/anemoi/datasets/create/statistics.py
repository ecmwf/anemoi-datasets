# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime

import numpy as np


class _Collector:
    def __init__(self) -> None:
        self._sum = np.float64(0.0)
        self._count = np.int64(0)
        self._min = np.float64(np.inf)
        self._max = -np.float64(np.inf)
        self._sumsq = np.float64(0.0)

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
            return {}

        mean = self._sum / self._count
        stdev = (self._sumsq / self._count) - (mean**2)
        return {
            "mean": mean,
            "minimum": self._min,
            "maximum": self._max,
            "stdev": stdev,
            "count": self._count,
        }


class StatisticsCollector:

    def __init__(self, cutoff_date: datetime.datetime | None = None) -> None:
        self.cutoff_date = cutoff_date

        self._collectors = None

    def collect(self, offset: int, array: any, dates: any) -> None:
        if not self.is_active(offset, array, dates):
            return

        if self._collectors is None:
            self._collectors = [_Collector() for _ in range(array.shape[1])]

        for i in range(array.shape[1]):
            self._collectors[i].update(array[:, i])

    def is_active(self, offset: int, array: any, dates: any) -> bool:
        return self.cutoff_date is None or dates[0] <= self.cutoff_date

    def statistics(self) -> list[dict[str, float]]:
        if self._collectors is None:
            return {}

        keys = list(self._collectors[0].statistics().keys())
        result = {key: [] for key in keys}
        for collector in self._collectors:
            stats = collector.statistics()
            for key in keys:
                result[key].append(stats[key])

        for key in keys:
            result[key] = np.array(result[key])

        return result


class NoStatisticsCollector(StatisticsCollector):
    def is_active(self, offset: int, array: any, dates: any) -> None:
        return False
