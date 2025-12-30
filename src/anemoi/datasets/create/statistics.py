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


class _Collector(_CollectorBase):
    pass


class _TendencyCollector(_CollectorBase):
    def __init__(self, column, name: str, delta: int) -> None:
        super().__init__(column)
        self._name = name
        self._delta = delta
        self._queue = []

    def update(self, data):
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

    def collect(self, array: any, dates: any, progress=_identity) -> None:

        if self._collectors is None:
            names = self._variables_names
            self._collectors = [_Collector(str(_) if names is None else names[_]) for _ in range(array.shape[1])]

            for name, delta in self._tendencies.items():
                self._tendencies_collectors[name] = [
                    _TendencyCollector(str(_) if names is None else names[_], name, delta)
                    for _ in range(array.shape[1])
                ]

        for i in progress(self._filter(dates)):

            data = array[i]

            # This part is negligeble compared to data access. No need to optimise.

            for j in range(array.shape[1]):
                values = data[j]
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

    def tendencies_statistics(self) -> dict[str, list[dict[str, float]]]:
        if self._tendencies_collectors is None:
            LOG.warning("No tendencies statistics collected")
            return {name: {_: np.array([np.nan]) for _ in STATISTICS} for name in self._tendencies.keys()}

        result = {}

        return result


def fix_variance(x: float, name: str, count: np.array, sums: np.array, squares: np.array) -> float:
    """Fix negative variance values due to numerical errors.

    Parameters
    ----------
    x : float
        The variance value.
    name : str
        The variable name.
    count : numpy.ndarray
        The count array.
    sums : numpy.ndarray
        The sums array.
    squares : numpy.ndarray
        The squares array.

    Returns
    -------
    float
        The fixed variance value.
    """
    assert count.shape == sums.shape == squares.shape
    assert isinstance(x, float)

    mean = sums / count
    assert mean.shape == count.shape

    if x >= 0:
        return x

    LOG.warning(f"Negative variance for {name=}, variance={x}")
    magnitude = np.sqrt((squares / count + mean * mean) / 2)
    LOG.warning(f"square / count - mean * mean =  {squares/count} - {mean*mean} = {squares/count - mean*mean}")
    LOG.warning(f"Variable span order of magnitude is {magnitude}.")
    LOG.warning(f"Count is {count}.")

    variances = squares / count - mean * mean
    assert variances.shape == squares.shape == mean.shape
    if np.all(variances >= 0):
        LOG.warning(f"All individual variances for {name} are positive, setting variance to 0.")
        return 0

    # if abs(x) < magnitude * 1e-6 and abs(x) < range * 1e-6:
    #     LOG.warning("Variance is negative but very small.")
    #     variances = squares / count - mean * mean
    #     return 0

    LOG.warning(f"ERROR at least one individual variance is negative ({np.nanmin(variances)}).")
    return 0


def check_variance(
    x: np.array,
    variables_names: list[str],
    minimum: np.array,
    maximum: np.array,
    mean: np.array,
    count: np.array,
    sums: np.array,
    squares: np.array,
) -> None:
    """Check for negative variance values and raise an error if found.

    Parameters
    ----------
    x : numpy.ndarray
        The variance array.
    variables_names : list of str
        List of variable names.
    minimum : numpy.ndarray
        The minimum values array.
    maximum : numpy.ndarray
        The maximum values array.
    mean : numpy.ndarray
        The mean values array.
    count : numpy.ndarray
        The count array.
    sums : numpy.ndarray
        The sums array.
    squares : numpy.ndarray
        The squares array.

    Raises
    ------
    ValueError
        If negative variance is found.
    """
    if (x >= 0).all():
        return
    print(x)
    print(variables_names)
    for i, (name, y) in enumerate(zip(variables_names, x)):
        if y >= 0:
            continue
        print("---")
        print(f"❗ Negative variance for {name=}, variance={y}")
        print(f" min={minimum[i]} max={maximum[i]} mean={mean[i]} count={count[i]} sums={sums[i]} squares={squares[i]}")
        print(f" -> sums: min={np.min(sums[i])}, max={np.max(sums[i])}, argmin={np.argmin(sums[i])}")
        print(f" -> squares: min={np.min(squares[i])}, max={np.max(squares[i])}, argmin={np.argmin(squares[i])}")
        print(f" -> count: min={np.min(count[i])}, max={np.max(count[i])}, argmin={np.argmin(count[i])}")
        print(
            f" squares / count - mean * mean =  {squares[i] / count[i]} - {mean[i] * mean[i]} = {squares[i] / count[i] - mean[i] * mean[i]}"
        )

    raise ValueError("Negative variance")
