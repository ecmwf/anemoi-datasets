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

import datetime
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.gridded.store import GriddedZarr

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
    any date is reproducible independently of how it is sliced.
    """

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


# --------------------------------------------------------------------------
# Grid resolvers
# --------------------------------------------------------------------------
def _latlon_from_npz(data: Any) -> tuple[NDArray[Any], NDArray[Any]]:
    """Extract latitude/longitude arrays from an ``.npz``-like mapping."""
    names = data.files if hasattr(data, "files") else list(data)
    keys = {k.lower(): k for k in names}
    lat_key = next((keys[k] for k in ("latitudes", "latitude", "lat") if k in keys), None)
    lon_key = next((keys[k] for k in ("longitudes", "longitude", "lon") if k in keys), None)
    if lat_key is None or lon_key is None:
        raise ValueError(f"grid data has no recognised latitude/longitude arrays: {list(keys.values())}")
    return np.asarray(data[lat_key], dtype=float), np.asarray(data[lon_key], dtype=float)


def _resolve_bbox(grid: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    bbox = grid["bbox"]
    if len(bbox) != 4:
        raise ValueError("synthetic 'bbox' grid must be [north, west, south, east]")
    if "resolution" not in grid:
        raise ValueError("synthetic 'bbox' grid requires a 'resolution'")
    north, west, south, east = (float(x) for x in bbox)
    if south > north:
        raise ValueError("synthetic 'bbox' grid: south must be <= north")
    if east < west:
        raise ValueError("synthetic 'bbox' grid: east must be >= west (antimeridian wrapping is not supported)")
    res = grid["resolution"]
    if isinstance(res, (list, tuple)):
        dlat, dlon = float(res[0]), float(res[1])
    else:
        dlat = dlon = float(res)
    if dlat <= 0 or dlon <= 0:
        raise ValueError("synthetic 'bbox' resolution must be positive")
    # ``linspace`` with a computed point count pins both endpoints exactly; a
    # float-step ``arange`` would leave rounding fuzz on the grid edges.
    n_lat = int(round((north - south) / dlat)) + 1
    n_lon = int(round((east - west) / dlon)) + 1
    lats_1d = np.linspace(north, south, n_lat)
    lons_1d = np.linspace(west, east, n_lon)
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    field_shape = lat_grid.shape  # (n_lat, n_lon)
    return lat_grid.reshape(-1), lon_grid.reshape(-1), field_shape


def _resolve_named(grid: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    from anemoi.transform.grids.named import lookup

    data = lookup(grid["named"])
    lat, lon = _latlon_from_npz(data)
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    return lat, lon, (lat.size,)


def _resolve_icon(grid: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    from anemoi.transform.grids.icon import IconGrid

    spec = grid["icon"]
    if isinstance(spec, str):
        spec = {"path": spec}
    if "path" not in spec:
        raise ValueError("synthetic 'icon' grid requires a 'path'")
    icon_grid = IconGrid(spec["path"], spec.get("refinement_level_c"))
    lat, lon = icon_grid.latlon()
    lat = np.asarray(lat, dtype=float).reshape(-1)
    lon = np.asarray(lon, dtype=float).reshape(-1)
    return lat, lon, (lat.size,)


def _resolve_unstructured(grid: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    spec = grid["unstructured"]
    if isinstance(spec, str):
        lat, lon = _latlon_from_npz(np.load(spec))
    else:
        lat, lon = _latlon_from_npz(spec)
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    if lat.shape != lon.shape:
        raise ValueError("synthetic 'unstructured' grid: latitudes and longitudes must have the same length")
    return lat, lon, (lat.size,)


_GRID_RESOLVERS = {
    "bbox": _resolve_bbox,
    "named": _resolve_named,
    "icon": _resolve_icon,
    "unstructured": _resolve_unstructured,
}


def resolve_grid(grid: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    """Resolve a ``grid`` spec to flat ``(latitudes, longitudes, field_shape)``."""
    if not isinstance(grid, dict):
        raise ValueError("synthetic 'grid' must be a dict")
    type_keys = [k for k in _GRID_RESOLVERS if k in grid]
    if len(type_keys) != 1:
        raise ValueError(
            f"synthetic 'grid' must contain exactly one of {sorted(_GRID_RESOLVERS)}; got keys {sorted(grid)}"
        )
    return _GRID_RESOLVERS[type_keys[0]](grid)


# --------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------
_KNOWN_KEYS = {"grid", "variables", "start", "end", "frequency", "values", "ensemble", "seed", "dtype", "resolution"}
_REQUIRED_KEYS = ("grid", "variables", "start", "end", "frequency")
_DEFAULT_VALUE_SPEC = {"mode": "random", "mean": 0.0, "std": 1.0}


@dataclass
class SyntheticConfig:
    """A fully resolved, validated synthetic-dataset configuration."""

    latitudes: NDArray[Any]
    longitudes: NDArray[Any]
    field_shape: Shape
    variables: list[str]
    dates: NDArray[np.datetime64]
    frequency: datetime.timedelta
    n_ensemble: int
    generators: list[ValueGenerator]  # aligned with ``variables``
    dtype: np.dtype
    resolution: str
    seed: int


def _build_variables(spec: Any) -> list[str]:
    if isinstance(spec, bool):  # bool is a subclass of int — reject explicitly
        raise ValueError("synthetic 'variables' must be a list of names or a positive integer")
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("synthetic 'variables' count must be a positive integer")
        width = max(2, len(str(spec - 1)))
        return [f"var_{i:0{width}d}" for i in range(spec)]
    if isinstance(spec, (list, tuple)):
        names = [str(v) for v in spec]
        if not names:
            raise ValueError("synthetic 'variables' list must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("synthetic 'variables' list contains duplicate names")
        return names
    raise ValueError("synthetic 'variables' must be a list of names or a positive integer")


def _build_dates(start: Any, end: Any, frequency: Any) -> tuple[NDArray[np.datetime64], datetime.timedelta]:
    freq = frequency_to_timedelta(frequency)
    step = np.timedelta64(freq)
    start = np.datetime64(start)
    end = np.datetime64(end)
    if end < start:
        raise ValueError("synthetic 'end' must not be before 'start'")
    n_steps = int((end - start) / step) + 1
    dates = (start + np.arange(n_steps) * step).astype("datetime64[s]")
    return dates, freq


def _resolve_resolution(raw: dict[str, Any]) -> str:
    if "resolution" in raw:
        return str(raw["resolution"])
    grid = raw["grid"]
    if "bbox" in grid and "resolution" in grid:
        return str(grid["resolution"])
    if "named" in grid:
        return str(grid["named"])
    return "unknown"


def _check_index_dtype(
    generators: list[ValueGenerator],
    dtype: np.dtype,
    n_dates: int,
    n_vars: int,
    n_ensemble: int,
    n_grid: int,
) -> None:
    """Reject an ``index`` value mode whose encoded positions overflow ``dtype``.

    :class:`IndexEncodedValue` encodes each cell's raveled position; that integer must
    be exactly representable in ``dtype`` or the no-shuffle guarantee silently breaks.
    """
    if not any(isinstance(g, IndexEncodedValue) for g in generators):
        return
    max_value = n_dates * n_vars * n_ensemble * n_grid - 1
    if np.issubdtype(dtype, np.floating):
        limit = 2 ** (np.finfo(dtype).nmant + 1)
    elif np.issubdtype(dtype, np.integer):
        limit = int(np.iinfo(dtype).max)
    else:
        return
    if max_value > limit:
        raise ValueError(
            f"synthetic 'index' value mode encodes positions up to {max_value}, which dtype "
            f"'{dtype}' cannot represent exactly; use a wider dtype such as float64"
        )


def _check_value_dtype(generators: list[ValueGenerator], dtype: np.dtype) -> None:
    """Reject an integer ``dtype`` for value modes that produce non-integer data.

    A ``random`` draw is continuous and a ``constant`` may be fractional; storing
    either in an integer array would silently truncate the values, leaving the
    analytic statistics disagreeing with the data actually returned.
    """
    if not np.issubdtype(dtype, np.integer):
        return
    for g in generators:
        if isinstance(g, RandomValue):
            raise ValueError(
                f"synthetic 'random' value mode produces non-integer values that integer "
                f"dtype '{dtype}' would truncate; use a floating-point dtype"
            )
        if isinstance(g, ConstantValue) and g.value != int(g.value):
            raise ValueError(
                f"synthetic 'constant' value {g.value} is not an integer and would be "
                f"truncated by integer dtype '{dtype}'; use a floating-point dtype"
            )


def parse_synthetic_config(raw: dict[str, Any]) -> SyntheticConfig:
    """Parse and validate the ``synthetic={...}`` argument into a :class:`SyntheticConfig`."""
    if not isinstance(raw, dict):
        raise ValueError("the 'synthetic' argument must be a dict")

    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown synthetic keys: {sorted(unknown)}; expected keys are {sorted(_KNOWN_KEYS)}")
    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise ValueError(f"synthetic config is missing required key {key!r}")

    latitudes, longitudes, field_shape = resolve_grid(raw["grid"])
    variables = _build_variables(raw["variables"])
    dates, frequency = _build_dates(raw["start"], raw["end"], raw["frequency"])

    n_ensemble = int(raw.get("ensemble", 1))
    if n_ensemble < 1:
        raise ValueError("synthetic 'ensemble' must be a positive integer")
    seed = int(raw.get("seed", 0))
    dtype = np.dtype(raw.get("dtype", "float32"))

    values = raw.get("values")
    if values is None:
        values = {}
    elif not isinstance(values, dict):
        raise ValueError("synthetic 'values' must be a dict")
    default_spec = values.get("default", _DEFAULT_VALUE_SPEC)
    generators = [build_value_generator(values.get(name, default_spec)) for name in variables]
    _check_index_dtype(generators, dtype, len(dates), len(variables), n_ensemble, int(latitudes.size))
    _check_value_dtype(generators, dtype)

    return SyntheticConfig(
        latitudes=latitudes,
        longitudes=longitudes,
        field_shape=field_shape,
        variables=variables,
        dates=dates,
        frequency=frequency,
        n_ensemble=n_ensemble,
        generators=generators,
        dtype=dtype,
        resolution=_resolve_resolution(raw),
        seed=seed,
    )


# --------------------------------------------------------------------------
# Lazy data
# --------------------------------------------------------------------------
_STAT_KEYS = ("mean", "stdev", "maximum", "minimum")


def _expand_index(index: tuple[Any, ...], ndim: int) -> tuple[Any, ...]:
    """Expand a single ``Ellipsis`` and right-pad with full slices to ``ndim`` axes."""
    # Identity checks, not ``in`` / ``count``: an element may be a numpy array,
    # for which ``== Ellipsis`` is an ambiguous elementwise comparison.
    ellipsis_at = [i for i, x in enumerate(index) if x is Ellipsis]
    if len(ellipsis_at) > 1:
        raise IndexError("Only one Ellipsis is allowed")
    if ellipsis_at:
        i = ellipsis_at[0]
        fill = ndim - (len(index) - 1)
        index = index[:i] + (slice(None),) * fill + index[i + 1 :]
    return index + (slice(None),) * (ndim - len(index))


class _SyntheticArray:
    """A lazy ``(dates, variables, ensemble, gridpoints)`` array.

    Values are synthesised when indexed, so only the dates actually requested are
    ever materialised -- the full array, which may be enormous for a long date
    range, is never built.
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config
        n_grid = int(config.latitudes.size)
        self.shape: Shape = (len(config.dates), len(config.variables), config.n_ensemble, n_grid)
        self.dtype = config.dtype
        self.ndim = 4
        self.chunks: Shape = (1, self.shape[1], self.shape[2], self.shape[3])

    def _generate(self, date_indices: NDArray[Any]) -> NDArray[Any]:
        """Synthesise ``(len(date_indices), variables, ensemble, gridpoints)``."""
        c = self._config
        _, n_vars, n_ensemble, n_grid = self.shape
        out = np.empty((len(date_indices), n_vars, n_ensemble, n_grid), dtype=c.dtype)
        for v, generator in enumerate(c.generators):
            out[:, v] = generator.generate(
                date_indices=date_indices,
                n_ensemble=n_ensemble,
                n_grid=n_grid,
                n_vars=n_vars,
                var_index=v,
                seed=c.seed,
            )
        return out

    def __getitem__(self, index: FullIndex) -> NDArray[Any]:
        index = _expand_index(index if isinstance(index, tuple) else (index,), self.ndim)
        axis0, rest = index[0], index[1:]

        # Delegate date-axis resolution to numpy: indexing ``arange(n_dates)``
        # reproduces numpy's bounds-checking, negative wrapping, and fancy /
        # boolean indexing -- so an out-of-range index raises ``IndexError``
        # instead of silently fabricating a nonexistent timestep.
        date_indices = np.arange(self.shape[0])[axis0]
        scalar = date_indices.ndim == 0
        block = self._generate(np.atleast_1d(date_indices))
        return block[(0, *rest)] if scalar else block[(slice(None), *rest)]


class _SyntheticStore:
    """A minimal lazy stand-in for the zarr group :class:`GriddedZarr` reads.

    It exposes exactly the surface :class:`GriddedZarr` consumes: item access for
    the data and the (small) coordinate/statistics arrays, plus an ``attrs`` dict.
    """

    def __init__(self, arrays: dict[str, Any], attrs: dict[str, Any]) -> None:
        self._arrays = arrays
        self.attrs = attrs

    def __getitem__(self, key: str) -> Any:
        return self._arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self._arrays


def _statistics_arrays(config: SyntheticConfig, method: str) -> dict[str, NDArray[Any]]:
    """Collect the per-variable statistics returned by ``method`` on each generator."""
    kwargs = dict(
        n_dates=len(config.dates),
        n_ensemble=config.n_ensemble,
        n_grid=int(config.latitudes.size),
        n_vars=len(config.variables),
        seed=config.seed,
    )
    per_variable = [getattr(g, method)(var_index=v, **kwargs) for v, g in enumerate(config.generators)]
    return {key: np.array([s[key] for s in per_variable], dtype=np.float64) for key in _STAT_KEYS}


def _build_synthetic_store(config: SyntheticConfig) -> _SyntheticStore:
    """Assemble the lazy store backing a :class:`SyntheticGriddedDataset`."""
    arrays: dict[str, Any] = {
        "data": _SyntheticArray(config),
        "dates": config.dates,
        "latitudes": config.latitudes,
        "longitudes": config.longitudes,
    }
    arrays.update(_statistics_arrays(config, "statistics"))

    # Tendency statistics need at least two dates; otherwise the keys stay absent
    # and statistics_tendencies() raises KeyError, as it would for a real dataset.
    if len(config.dates) >= 2:
        delta = frequency_to_string(config.frequency)
        for key, values in _statistics_arrays(config, "tendency_statistics").items():
            arrays[f"statistics_tendencies_{delta}_{key}"] = values

    attrs = {
        "layout": "gridded",
        "resolution": config.resolution,
        "frequency": frequency_to_string(config.frequency),
        "variables": list(config.variables),
        "variables_metadata": {v: {} for v in config.variables},
        "field_shape": [int(x) for x in config.field_shape],
        # Suppresses GriddedZarr's "no constant_fields" warning; it recomputes the value regardless.
        "constant_fields": [v for v, g in zip(config.variables, config.generators) if g.is_constant],
        "synthetic": True,
    }
    return _SyntheticStore(arrays, attrs)


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------
class SyntheticGriddedDataset(GriddedZarr):
    """An in-memory synthetic gridded dataset built from a :class:`SyntheticConfig`.

    It subclasses :class:`GriddedZarr` over a lazy store, so the whole dataset
    contract is inherited from the implementation that backs real datasets; only
    synthetic-aware presentation (repr, tree, metadata marker) and the analytic
    ``constant_fields`` are overridden. Nothing is materialised until the data is
    indexed.
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config
        super().__init__(_build_synthetic_store(config), path="synthetic")

    def __repr__(self) -> str:
        c = self._config
        return (
            f"SyntheticGriddedDataset(variables={len(c.variables)}, "
            f"dates={len(c.dates)}, gridpoints={int(c.latitudes.size)})"
        )

    def tree(self) -> Node:
        c = self._config
        return Node(
            self,
            [],
            synthetic=dict(
                variables=len(c.variables),
                dates=len(c.dates),
                gridpoints=int(c.latitudes.size),
            ),
        )

    @property
    def constant_fields(self) -> list[str]:
        # GriddedZarr.constant_fields always recomputes the value from the data,
        # which mislabels every field as constant when there is only one date.
        # The synthetic store already records the answer known analytically.
        return list(self.store.attrs["constant_fields"])

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        return {**super().metadata_specific(**kwargs), "synthetic": True}


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def synthetic_factory(args: tuple[Any, ...], kwargs: dict[str, Any]) -> SyntheticGriddedDataset:
    """Build a :class:`SyntheticGriddedDataset` from ``open_dataset(synthetic=...)``.

    The synthetic dataset is fully described by the ``synthetic`` dict; it does not
    compose with other ``open_dataset`` keywords.
    """
    assert len(args) == 0, args
    config = parse_synthetic_config(kwargs.pop("synthetic"))
    if kwargs:
        raise ValueError(
            f"open_dataset(synthetic=...) does not support extra keywords {sorted(kwargs)}; "
            "configure the dataset entirely inside the 'synthetic' dict"
        )
    return SyntheticGriddedDataset(config)
