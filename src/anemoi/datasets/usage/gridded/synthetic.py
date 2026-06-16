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


# --------------------------------------------------------------------------
# Computed forcings
# --------------------------------------------------------------------------
# Computed forcings reuse earthkit's canonical forcing formulas rather than
# reimplementing them: ``from_source("forcings", latitudes=, longitudes=, ...)``
# accepts raw lat/lon arrays with no template field, exactly fitting the
# synthetic dataset's flat grid.
_COMPUTED_FORCINGS = {
    "latitude",
    "longitude",
    "cos_latitude",
    "sin_latitude",
    "cos_longitude",
    "sin_longitude",
    "cos_julian_day",
    "sin_julian_day",
    "cos_local_time",
    "sin_local_time",
    "insolation",
    "cos_solar_zenith_angle",
    "toa_incident_solar_radiation",
    "ecef_x",
    "ecef_y",
    "ecef_z",
}

# Forcings whose value depends only on position, not on the valid time.
_TIME_INVARIANT_FORCINGS = {
    "latitude",
    "longitude",
    "cos_latitude",
    "sin_latitude",
    "cos_longitude",
    "sin_longitude",
    "ecef_x",
    "ecef_y",
    "ecef_z",
}

# Time-varying forcings have no closed-form statistics over an arbitrary date
# range, so the generator evaluates the field on up to this many evenly-spaced
# dates to estimate them. Opening stays cheap; the data path is never affected.
_FORCING_STATS_MAX_SAMPLES = 128


def _forcing_base(name: str) -> str:
    """Strip an earthkit time-delta suffix (``insolation+6h`` -> ``insolation``)."""
    for sep in ("+", "-"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name


def is_computed_forcing(name: str) -> bool:
    """Whether ``name`` names an earthkit computed forcing (delta suffixes allowed)."""
    return _forcing_base(name) in _COMPUTED_FORCINGS


class ComputedForcingValue(ValueGenerator):
    """A computed forcing (``insolation``, ``cos_latitude``, ...) evaluated through
    earthkit's forcings source from the dataset's own grid and dates.

    Time-invariant forcings (latitude/longitude-derived) are flagged
    :attr:`is_constant`; time-varying ones estimate their statistics from a sample
    of dates (see :data:`_FORCING_STATS_MAX_SAMPLES`).
    """

    def __init__(
        self,
        param: str,
        *,
        latitudes: NDArray[Any],
        longitudes: NDArray[Any],
        dates: NDArray[np.datetime64],
    ) -> None:
        self.param = param
        self.latitudes = np.asarray(latitudes, dtype=float)
        self.longitudes = np.asarray(longitudes, dtype=float)
        self.dates = dates
        self.is_constant = _forcing_base(param) in _TIME_INVARIANT_FORCINGS

    def _evaluate(self, dates: NDArray[np.datetime64]) -> NDArray[Any]:
        """Return a ``(len(dates), n_grid)`` block of the forcing field."""
        from earthkit.data import from_source

        py_dates = [np.datetime64(d, "s").astype(datetime.datetime) for d in dates]
        fields = from_source(
            "forcings",
            latitudes=self.latitudes,
            longitudes=self.longitudes,
            date=py_dates,
            param=[self.param],
        )
        # One field per date (single param, no ensemble), in date order.
        assert len(fields) == len(py_dates), (len(fields), len(py_dates))
        return np.stack([f.to_numpy(flatten=True) for f in fields]).astype(np.float64)

    def generate(self, *, date_indices, n_ensemble, n_grid, n_vars, var_index, seed):
        dates = self.dates[np.asarray(date_indices, dtype=int)]
        block = self._evaluate(dates)  # (len(dates), n_grid)
        out = np.empty((len(dates), n_ensemble, n_grid), dtype=np.float64)
        out[:] = block[:, None, :]  # forcings do not vary across ensemble members
        return out

    def _sample_indices(self, n_dates: int) -> NDArray[Any]:
        if n_dates <= _FORCING_STATS_MAX_SAMPLES:
            return np.arange(n_dates)
        return np.unique(np.linspace(0, n_dates - 1, _FORCING_STATS_MAX_SAMPLES).round().astype(int))

    def statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        block = self._evaluate(self.dates[self._sample_indices(n_dates)])
        return dict(
            mean=float(block.mean()),
            stdev=float(block.std()),
            maximum=float(block.max()),
            minimum=float(block.min()),
        )

    def tendency_statistics(self, *, n_dates, n_ensemble, n_grid, n_vars, var_index, seed):
        idx = self._sample_indices(n_dates)
        idx = idx[idx < n_dates - 1]  # need a successor date for each sampled step
        if idx.size == 0:
            idx = np.array([0])
        diff = self._evaluate(self.dates[idx + 1]) - self._evaluate(self.dates[idx])
        return dict(
            mean=float(diff.mean()),
            stdev=float(diff.std()),
            maximum=float(diff.max()),
            minimum=float(diff.min()),
        )


_VALUE_TYPES = ("constant", "random")


def _generator_from_type(type_name: str, payload: Any) -> ValueGenerator:
    if type_name == "constant":
        if isinstance(payload, bool) or not isinstance(payload, (int, float)):
            raise ValueError("synthetic 'constant' value must be a number given directly, e.g. {'constant': 273.15}")
        return ConstantValue(payload)
    if type_name == "random":
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("synthetic 'random' value spec must be a dict of parameters")
        return RandomValue(payload.get("mean", 0.0), payload.get("std", 1.0))
    raise ValueError(f"unknown synthetic value type {type_name!r}; expected one of {list(_VALUE_TYPES)}")


def build_value_generator(spec: Any) -> ValueGenerator:
    """Build a :class:`ValueGenerator` from a ``values`` config entry.

    Accepts a one-of type-key dict (``{"random": {...}}``, ``{"constant": 273.15}``),
    a bare scalar (shorthand for ``constant``), or a bare string (a generator name
    with default parameters).
    """
    if isinstance(spec, bool):  # bool is a subclass of int -- reject before the scalar branch
        raise ValueError("synthetic value spec must be a number, a string, or a one-of dict")
    if isinstance(spec, (int, float)):
        return ConstantValue(spec)
    if isinstance(spec, str):
        return _generator_from_type(spec, None)
    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError(f"synthetic value spec must have exactly one of {list(_VALUE_TYPES)}; got {sorted(spec)}")
        ((type_name, payload),) = spec.items()
        return _generator_from_type(type_name, payload)
    raise ValueError("synthetic value spec must be a number, a string, or a one-of dict")


# --------------------------------------------------------------------------
# Geography resolvers
# --------------------------------------------------------------------------
def _latlon_from_npz(data: Any) -> tuple[NDArray[Any], NDArray[Any]]:
    """Extract latitude/longitude arrays from an ``.npz``-like mapping."""
    names = data.files if hasattr(data, "files") else list(data)
    keys = {k.lower(): k for k in names}
    lat_key = next((keys[k] for k in ("latitudes", "latitude", "lat") if k in keys), None)
    lon_key = next((keys[k] for k in ("longitudes", "longitude", "lon") if k in keys), None)
    if lat_key is None or lon_key is None:
        raise ValueError(f"geography data has no recognised latitude/longitude arrays: {list(keys.values())}")
    return np.asarray(data[lat_key], dtype=float), np.asarray(data[lon_key], dtype=float)


def _resolve_bbox(geography: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    bbox = geography["bbox"]
    if len(bbox) != 4:
        raise ValueError("synthetic 'bbox' geography must be [north, west, south, east]")
    if "resolution" not in geography:
        raise ValueError("synthetic 'bbox' geography requires a 'resolution'")
    north, west, south, east = (float(x) for x in bbox)
    if south > north:
        raise ValueError("synthetic 'bbox' geography: south must be <= north")
    if east < west:
        raise ValueError("synthetic 'bbox' geography: east must be >= west (antimeridian wrapping is not supported)")
    res = geography["resolution"]
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


def _resolve_named(geography: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    from anemoi.transform.grids.named import lookup

    data = lookup(geography["named"])
    lat, lon = _latlon_from_npz(data)
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    return lat, lon, (lat.size,)


def _resolve_icon(geography: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    from anemoi.transform.grids.icon import IconGrid

    spec = geography["icon"]
    if isinstance(spec, str):
        spec = {"path": spec}
    if "path" not in spec:
        raise ValueError("synthetic 'icon' geography requires a 'path'")
    icon_grid = IconGrid(spec["path"], spec.get("refinement_level_c"))
    lat, lon = icon_grid.latlon()
    lat = np.asarray(lat, dtype=float).reshape(-1)
    lon = np.asarray(lon, dtype=float).reshape(-1)
    return lat, lon, (lat.size,)


def _resolve_unstructured(geography: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    spec = geography["unstructured"]
    if isinstance(spec, str):
        lat, lon = _latlon_from_npz(np.load(spec))
    else:
        lat, lon = _latlon_from_npz(spec)
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    if lat.shape != lon.shape:
        raise ValueError("synthetic 'unstructured' geography: latitudes and longitudes must have the same length")
    return lat, lon, (lat.size,)


_GEOGRAPHY_RESOLVERS = {
    "bbox": _resolve_bbox,
    "named": _resolve_named,
    "icon": _resolve_icon,
    "unstructured": _resolve_unstructured,
}


def resolve_geography(geography: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], Shape]:
    """Resolve a ``geography`` spec to flat ``(latitudes, longitudes, field_shape)``."""
    if not isinstance(geography, dict):
        raise ValueError("synthetic 'geography' must be a dict")
    type_keys = [k for k in _GEOGRAPHY_RESOLVERS if k in geography]
    if len(type_keys) != 1:
        raise ValueError(
            f"synthetic 'geography' must contain exactly one of {sorted(_GEOGRAPHY_RESOLVERS)}; "
            f"got keys {sorted(geography)}"
        )
    return _GEOGRAPHY_RESOLVERS[type_keys[0]](geography)


# --------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------
_KNOWN_KEYS = {"geography", "variables", "dates", "layout", "values", "ensembles", "seed", "dtype", "resolution"}
_REQUIRED_KEYS = ("geography", "variables", "dates", "layout")
_DATE_KEYS = ("start", "end", "frequency")
_VARIABLE_KEYS = {"name", "metadata", "values", "statistics", "tendencies_statistics"}
_LAYOUTS = ("gridded", "tabular", "trajectories")
_DEFAULT_VALUE_SPEC = {"random": {"mean": 0.0, "std": 1.0}}
_STAT_KEYS = ("mean", "stdev", "maximum", "minimum")


@dataclass
class _Variable:
    """One parsed ``variables`` entry, before generators are built."""

    name: str
    metadata: dict[str, Any]
    values: Any  # raw value spec, or None to use the dataset default
    statistics: dict[str, float] | None
    tendencies_statistics: dict[str, float] | None


@dataclass
class SyntheticConfig:
    """A fully resolved, validated synthetic-dataset configuration."""

    latitudes: NDArray[Any]
    longitudes: NDArray[Any]
    field_shape: Shape
    variables: list[str]
    variables_metadata: dict[str, dict[str, Any]]
    dates: NDArray[np.datetime64]
    frequency: datetime.timedelta
    n_ensemble: int
    generators: list[ValueGenerator]  # aligned with ``variables``
    stats_overrides: list[dict[str, float] | None]  # aligned with ``variables``
    tendency_overrides: list[dict[str, float] | None]  # aligned with ``variables``
    layout: str
    dtype: np.dtype
    resolution: str
    seed: int


def _validate_stats_override(override: Any, label: str) -> dict[str, float] | None:
    if override is None:
        return None
    if not isinstance(override, dict):
        raise ValueError(f"synthetic '{label}' must be a dict of statistics")
    unknown = set(override) - set(_STAT_KEYS)
    if unknown:
        raise ValueError(f"unknown synthetic '{label}' keys: {sorted(unknown)}; expected {sorted(_STAT_KEYS)}")
    return {k: float(v) for k, v in override.items()}


def _parse_variable_entry(entry: Any) -> _Variable:
    if isinstance(entry, str):
        return _Variable(name=entry, metadata={}, values=None, statistics=None, tendencies_statistics=None)
    if not isinstance(entry, dict):
        raise ValueError(f"synthetic 'variables' entry must be a string or a dict, got {type(entry).__name__}")
    unknown = set(entry) - _VARIABLE_KEYS
    if unknown:
        raise ValueError(f"unknown synthetic variable keys: {sorted(unknown)}; expected {sorted(_VARIABLE_KEYS)}")
    if "name" not in entry:
        raise ValueError("synthetic 'variables' dict entry is missing required key 'name'")
    name = str(entry["name"])
    metadata = entry.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"synthetic variable '{name}' metadata must be a dict")
    return _Variable(
        name=name,
        metadata=dict(metadata),
        values=entry.get("values"),
        statistics=_validate_stats_override(entry.get("statistics"), "statistics"),
        tendencies_statistics=_validate_stats_override(entry.get("tendencies_statistics"), "tendencies_statistics"),
    )


def _build_variables(spec: Any) -> list[_Variable]:
    if not isinstance(spec, (list, tuple)):
        raise ValueError("synthetic 'variables' must be a list of names or dicts")
    if not spec:
        raise ValueError("synthetic 'variables' list must be non-empty")
    variables = [_parse_variable_entry(e) for e in spec]
    names = [v.name for v in variables]
    if len(set(names)) != len(names):
        raise ValueError("synthetic 'variables' list contains duplicate names")
    return variables


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
    geography = raw["geography"]
    if "bbox" in geography and "resolution" in geography:
        return str(geography["resolution"])
    if "named" in geography:
        return str(geography["named"])
    return "unknown"


def _forcing_metadata(name: str) -> dict[str, Any]:
    return {"computed_forcing": True, "constant_in_time": _forcing_base(name) in _TIME_INVARIANT_FORCINGS}


def _build_generator(
    variable: _Variable,
    default_spec: Any,
    *,
    latitudes: NDArray[Any],
    longitudes: NDArray[Any],
    dates: NDArray[np.datetime64],
) -> ValueGenerator:
    if is_computed_forcing(variable.name):
        if variable.values is not None:
            raise ValueError(
                f"synthetic variable '{variable.name}' is a computed forcing and cannot take a 'values' block; "
                "it generates its own values"
            )
        return ComputedForcingValue(variable.name, latitudes=latitudes, longitudes=longitudes, dates=dates)
    spec = variable.values if variable.values is not None else default_spec
    return build_value_generator(spec)


def _check_value_dtype(generators: list[ValueGenerator], dtype: np.dtype) -> None:
    """Reject an integer ``dtype`` for generators that produce non-integer data.

    A ``random`` draw, a computed forcing, or a fractional ``constant`` stored in
    an integer array would silently truncate, leaving the analytic statistics
    disagreeing with the data actually returned.
    """
    if not np.issubdtype(dtype, np.integer):
        return
    for g in generators:
        if isinstance(g, (RandomValue, ComputedForcingValue)):
            kind = "random" if isinstance(g, RandomValue) else f"computed forcing '{g.param}'"
            raise ValueError(
                f"synthetic '{kind}' produces non-integer values that integer dtype '{dtype}' would truncate; "
                "use a floating-point dtype"
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

    layout = raw["layout"]
    if layout not in _LAYOUTS:
        raise ValueError(f"synthetic 'layout' must be one of {list(_LAYOUTS)}; got {layout!r}")
    if layout != "gridded":
        raise NotImplementedError(f"synthetic '{layout}' layout is not implemented yet; only 'gridded' is supported")

    latitudes, longitudes, field_shape = resolve_geography(raw["geography"])
    variables = _build_variables(raw["variables"])

    date_spec = raw["dates"]
    if not isinstance(date_spec, dict):
        raise ValueError("synthetic 'dates' must be a dict with 'start', 'end' and 'frequency'")
    unknown_dates = set(date_spec) - set(_DATE_KEYS)
    if unknown_dates:
        raise ValueError(f"unknown synthetic 'dates' keys: {sorted(unknown_dates)}; expected {sorted(_DATE_KEYS)}")
    for key in _DATE_KEYS:
        if key not in date_spec:
            raise ValueError(f"synthetic 'dates' is missing required key {key!r}")
    dates, frequency = _build_dates(date_spec["start"], date_spec["end"], date_spec["frequency"])

    n_ensemble = int(raw.get("ensembles", 1))
    if n_ensemble < 1:
        raise ValueError("synthetic 'ensembles' must be a positive integer")
    seed = int(raw.get("seed", 0))
    dtype = np.dtype(raw.get("dtype", "float32"))

    default_spec = raw.get("values", _DEFAULT_VALUE_SPEC)

    generators = [
        _build_generator(v, default_spec, latitudes=latitudes, longitudes=longitudes, dates=dates) for v in variables
    ]
    _check_value_dtype(generators, dtype)

    variables_metadata = {}
    for v in variables:
        meta = _forcing_metadata(v.name) if is_computed_forcing(v.name) else {}
        meta.update(v.metadata)  # explicit metadata wins over the auto-derived forcing flags
        variables_metadata[v.name] = meta

    return SyntheticConfig(
        latitudes=latitudes,
        longitudes=longitudes,
        field_shape=field_shape,
        variables=[v.name for v in variables],
        variables_metadata=variables_metadata,
        dates=dates,
        frequency=frequency,
        n_ensemble=n_ensemble,
        generators=generators,
        stats_overrides=[v.statistics for v in variables],
        tendency_overrides=[v.tendencies_statistics for v in variables],
        layout=layout,
        dtype=dtype,
        resolution=_resolve_resolution(raw),
        seed=seed,
    )


# --------------------------------------------------------------------------
# Lazy data
# --------------------------------------------------------------------------
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
    """Collect the per-variable statistics returned by ``method`` on each generator,
    applying any per-variable override.
    """
    kwargs = dict(
        n_dates=len(config.dates),
        n_ensemble=config.n_ensemble,
        n_grid=int(config.latitudes.size),
        n_vars=len(config.variables),
        seed=config.seed,
    )
    overrides = config.stats_overrides if method == "statistics" else config.tendency_overrides
    per_variable = []
    for v, g in enumerate(config.generators):
        stats = getattr(g, method)(var_index=v, **kwargs)
        if overrides[v]:
            stats = {**stats, **overrides[v]}
        per_variable.append(stats)
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
        "layout": config.layout,
        "resolution": config.resolution,
        "frequency": frequency_to_string(config.frequency),
        "variables": list(config.variables),
        "variables_metadata": dict(config.variables_metadata),
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
    """Build a :class:`SyntheticGriddedDataset` from the ``synthetic`` argument.

    Only the ``synthetic`` keyword is consumed here; any remaining transform
    keywords (``select``, ``start``, ``rename``, ...) are left in ``kwargs`` for
    the caller to apply via :meth:`Dataset._subset`, exactly as for ``dataset=``.
    """
    assert len(args) == 0, args
    config = parse_synthetic_config(kwargs.pop("synthetic"))
    return SyntheticGriddedDataset(config)
