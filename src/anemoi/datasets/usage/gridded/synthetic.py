# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Synthetic gridded dataset, opened with ``open_dataset(synthetic={...})``.

Data is generated lazily on indexing, so a dataset spanning any date range can be
opened without building or storing the full array. Intended for testing and
prototyping anemoi training/inference pipelines without a Zarr store on disk.
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Value generators
# --------------------------------------------------------------------------
# A normally distributed field has no true min/max; the synthetic dataset reports
# its extrema as ``mean +/- _RANDOM_EXTREMUM_SIGMA * stdev``.
_RANDOM_EXTREMUM_SIGMA = 5.0


class ValueGenerator(ABC):
    """Produces the synthetic values for one variable.

    Generators are pure functions of position: :meth:`generate` synthesises an
    arbitrary set of dates on demand (so the dataset is never materialised in
    full), and :meth:`statistics` / :meth:`tendency_statistics` report the
    field's statistics in closed form (so opening a dataset costs O(1)).
    """

    is_constant: bool = False

    @abstractmethod
    def generate(
        self,
        *,
        date_indices: NDArray[Any],
        n_ensemble: int,
        n_grid: int,
        n_vars: int,
        var_index: int,
        seed: int,
    ) -> NDArray[Any]:
        """Return a ``(len(date_indices), n_ensemble, n_grid)`` float64 array."""

    @abstractmethod
    def statistics(
        self, *, n_dates: int, n_ensemble: int, n_grid: int, n_vars: int, var_index: int, seed: int
    ) -> dict[str, float]:
        """Return the analytic ``mean``/``stdev``/``maximum``/``minimum`` of the field."""

    @abstractmethod
    def tendency_statistics(
        self, *, n_dates: int, n_ensemble: int, n_grid: int, n_vars: int, var_index: int, seed: int
    ) -> dict[str, float]:
        """Return the analytic statistics of the one-step (date-to-date) tendency."""


class ConstantValue(ValueGenerator):
    """Every value equals a fixed constant."""

    is_constant = True

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def generate(self, *, date_indices, n_ensemble, n_grid, n_vars, var_index, seed):
        return np.full((len(date_indices), n_ensemble, n_grid), self.value, dtype=np.float64)

    def statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        return dict(mean=self.value, stdev=0.0, maximum=self.value, minimum=self.value)

    def tendency_statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        return dict(mean=0.0, stdev=0.0, maximum=0.0, minimum=0.0)


class RandomValue(ValueGenerator):
    """Gaussian noise. Each date is seeded by ``(seed, var_index, date_index)``, so
    any date is reproducible independently of how it is sliced."""

    is_constant = False

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = float(mean)
        self.std = float(std)

    def generate(self, *, date_indices, n_ensemble, n_grid, n_vars, var_index, seed):
        out = np.empty((len(date_indices), n_ensemble, n_grid), dtype=np.float64)
        for i, date_index in enumerate(date_indices):
            rng = np.random.default_rng((seed, var_index, int(date_index)))
            out[i] = rng.normal(self.mean, self.std, size=(n_ensemble, n_grid))
        return out

    def statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        spread = _RANDOM_EXTREMUM_SIGMA * self.std
        return dict(mean=self.mean, stdev=self.std, maximum=self.mean + spread, minimum=self.mean - spread)

    def tendency_statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        # The difference of two independent N(mean, std**2) draws is N(0, 2 std**2).
        tendency_std = self.std * np.sqrt(2.0)
        spread = _RANDOM_EXTREMUM_SIGMA * tendency_std
        return dict(mean=0.0, stdev=tendency_std, maximum=spread, minimum=-spread)


class IndexEncodedValue(ValueGenerator):
    """Each value encodes its own raveled ``(date, variable, ensemble, gridpoint)``
    position: ``((date * n_vars + var) * n_ensemble + ens) * n_grid + grid``.

    Lets a test assert that a pipeline did not shuffle or misalign data.
    """

    is_constant = False

    def generate(self, *, date_indices, n_ensemble, n_grid, n_vars, var_index, seed):
        d = np.asarray(date_indices, dtype=np.int64).reshape(-1, 1, 1)
        e = np.arange(n_ensemble).reshape(1, n_ensemble, 1)
        g = np.arange(n_grid).reshape(1, 1, n_grid)
        return (((d * n_vars + var_index) * n_ensemble + e) * n_grid + g).astype(np.float64)

    def statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        # value = a*date + b*ensemble + grid + c, with date/ensemble/grid uniform.
        a = n_vars * n_ensemble * n_grid
        b = n_grid
        c = var_index * n_ensemble * n_grid
        mean = a * (n_dates - 1) / 2 + b * (n_ensemble - 1) / 2 + (n_grid - 1) / 2 + c
        # The variance of a discrete uniform on 0..n-1 is (n**2 - 1) / 12; the three axes are independent.
        var = a**2 * (n_dates**2 - 1) / 12 + b**2 * (n_ensemble**2 - 1) / 12 + (n_grid**2 - 1) / 12
        maximum = a * (n_dates - 1) + b * (n_ensemble - 1) + (n_grid - 1) + c
        return dict(mean=float(mean), stdev=float(np.sqrt(var)), maximum=float(maximum), minimum=float(c))

    def tendency_statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        # value(date + 1) - value(date) is the constant a = n_vars * n_ensemble * n_grid.
        a = float(n_vars * n_ensemble * n_grid)
        return dict(mean=a, stdev=0.0, maximum=a, minimum=a)


def build_value_generator(spec: dict[str, Any]) -> ValueGenerator:
    """Build a :class:`ValueGenerator` from a ``values`` config entry."""
    if not isinstance(spec, dict):
        raise ValueError(f"synthetic value spec must be a dict, got {type(spec).__name__}")
    mode = spec.get("mode")
    if mode is None:
        raise ValueError("synthetic value spec is missing required key 'mode'")
    if mode == "constant":
        if "value" not in spec:
            raise ValueError("synthetic 'constant' value spec requires a 'value'")
        return ConstantValue(spec["value"])
    if mode == "random":
        return RandomValue(spec.get("mean", 0.0), spec.get("std", 1.0))
    if mode == "index":
        return IndexEncodedValue()
    raise ValueError(f"Unknown synthetic value mode {mode!r}; expected 'constant', 'random' or 'index'")
