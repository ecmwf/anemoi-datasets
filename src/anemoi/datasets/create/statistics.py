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


class _CollectorBase:
    def __init__(self, num_columns, column_names=None) -> None:
        self._count = np.zeros(num_columns, dtype=np.int64)
        self._min = np.full(num_columns, np.inf, dtype=np.float64)
        self._max = np.full(num_columns, -np.inf, dtype=np.float64)
        self._column_names = column_names or [str(i) for i in range(num_columns)]
        self._mean = np.zeros(num_columns, dtype=np.float64)
        self._m2 = np.zeros(num_columns, dtype=np.float64)

    def update(self, data: any) -> None:
        # data shape: (n_samples, n_columns)
        data = data.astype(np.float64)

        # Create mask for valid data (not NaN)
        valid_mask = ~np.isnan(data)

        # Process each column
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            col_mask = valid_mask[:, col_idx]
            valid_data = col_data[col_mask]

            if valid_data.size == 0:
                continue

            n = len(valid_data)
            old_mean = self._mean[col_idx]
            old_count = self._count[col_idx]
            new_count = old_count + n

            # Batch mean
            batch_mean = np.mean(valid_data)

            # Update combined mean
            self._mean[col_idx] = (old_count * old_mean + n * batch_mean) / new_count

            # Update M2 using parallel algorithm formula
            batch_m2 = np.sum((valid_data - batch_mean) ** 2)
            delta = batch_mean - old_mean
            self._m2[col_idx] = self._m2[col_idx] + batch_m2 + (old_count * n * delta * delta) / new_count

            self._count[col_idx] = new_count

            # Update min/max
            self._min[col_idx] = min(self._min[col_idx], np.min(valid_data))
            self._max[col_idx] = max(self._max[col_idx], np.max(valid_data))

    def statistics(self) -> dict[str, np.ndarray]:
        # Returns arrays for each statistic across all columns
        result = {}

        for col_idx in range(len(self._count)):
            if self._count[col_idx] == 0:
                LOG.warning(f"Column {self._column_names[col_idx]}: no statistics collected")

        # Compute variance using Welford's M2
        variance = np.where(self._count > 0, self._m2 / self._count, 0.0)

        # Check for negative variance (numerical errors)
        negative_mask = variance < 0
        if np.any(negative_mask):
            for col_idx in np.where(negative_mask)[0]:
                var = variance[col_idx]
                relative_error = abs(var) / max(abs(self._m2[col_idx]), abs(self._mean[col_idx] ** 2), 1e-100)
                if relative_error > 1e-6:
                    msg = f"Negative variance {var} for column {self._column_names[col_idx]}, {relative_error=}"
                    msg += f" m2={self._m2[col_idx]}, count={self._count[col_idx]}, mean={self._mean[col_idx]}"
                    msg += f", min={self._min[col_idx]}, max={self._max[col_idx]}"
                    LOG.error(msg)
                    raise ValueError(msg)
                variance[col_idx] = 0.0

        stdev = np.sqrt(variance)

        # Set NaN for columns with no data
        no_data_mask = self._count == 0
        result = {
            "mean": np.where(no_data_mask, np.nan, self._mean),
            "minimum": np.where(no_data_mask, np.nan, self._min),
            "maximum": np.where(no_data_mask, np.nan, self._max),
            "stdev": np.where(no_data_mask, np.nan, stdev),
        }

        return result


class _Collector(_CollectorBase):
    pass


class _TendencyCollector(_CollectorBase):
    def __init__(self, num_columns, column_names, name: str, delta: int) -> None:
        super().__init__(num_columns, column_names)
        self._name = name
        self._delta = delta
        # Only keep a sliding window of the last 'delta' rows
        self._window = None
        self._window_size = 0

    def update(self, data):
        data = data.astype(np.float64)

        # Concatenate window with new data for tendency computation
        if self._window is None:
            combined = data
        else:
            combined = np.concatenate([self._window, data], axis=0)

        # Compute tendencies wherever we have enough history
        if len(combined) > self._delta:
            tendencies = combined[self._delta :] - combined[: -self._delta]
            super().update(tendencies)

        # Update sliding window: keep only last 'delta' rows
        self._window = combined[-self._delta :].copy()
        self._window_size = len(self._window)

    def finalize(self):
        """Nothing to do - all processing done in update()"""
        pass


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

        self._collector = None
        self._variables_names = variables_names
        self._allow_nans = allow_nans
        self._tendencies = tendencies or {}
        self._tendencies_collectors = {}

    def collect(self, array: any, dates: any) -> None:

        if self._collector is None:
            num_columns = array.shape[1]
            names = self._variables_names
            column_names = [str(i) if names is None else names[i] for i in range(num_columns)]

            # Single collector for all columns
            self._collector = _Collector(num_columns, column_names)

            # Tendency collectors
            for name, delta in self._tendencies.items():
                self._tendencies_collectors[name] = _TendencyCollector(num_columns, column_names, name, delta)

        # Update all columns at once
        self._collector.update(array)

        # For tendencies, just buffer the data
        for c in self._tendencies_collectors.values():
            c.update(array)

    def statistics(self) -> dict[str, np.ndarray]:
        if self._collector is None:
            LOG.warning("No statistics collected")
            return {_: np.array([np.nan]) for _ in STATISTICS}

        result = self._collector.statistics()

        # Finalize tendency collectors before getting statistics
        for name, collector in self._tendencies_collectors.items():
            collector.finalize()
            tendencies = collector.statistics()

            for key in STATISTICS:
                result[f"statistics_tendencies_{name}_{key}"] = tendencies[key]

        return result
