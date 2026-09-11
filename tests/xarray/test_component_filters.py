# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for filters that combine several fields, over an Xarray source.

``r-to-q``, ``r-to-d`` and ``uv-to-ddff`` each build one field out of two, so
they must first work out which fields belong together.  ``anemoi.transform``'s
``GroupByParam`` does that by the ``mars`` metadata namespace, falling back to
the *whole* metadata dictionary when that namespace is empty.  The fallback
includes ``units``, which necessarily differs between the components being
combined (``K`` for a temperature, ``1`` for a relative humidity), so every
component lands in a group of its own and the filter fails with::

    ValueError: Missing component. Want ['r', 't'], got ['r']

These tests run the filters through a real recipe over a real ``xarray-zarr``
store, which is the combination that broke: nothing else in the test suite
pairs an Xarray source with a component filter.
"""

import datetime

import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr

N_LAT, N_LON = 3, 4
START = datetime.datetime(2021, 1, 1, 0)
N_TIMES = 2
FREQUENCY_H = 6
LEVELS = (500, 850, 1000)


def _run_one_group(recipe: dict) -> ekd.FieldList:
    """Build one group of a recipe and return the resulting fields.

    Parameters
    ----------
    recipe : dict
        The recipe to run.

    Returns
    -------
    ekd.FieldList
        The fields produced by the recipe's input pipeline.
    """
    from anemoi.datasets.create.input.builder import InputBuilder
    from anemoi.datasets.create.input.context import Context
    from anemoi.datasets.create.recipe import Recipe
    from anemoi.datasets.dates.groups import Groups

    class TestContext(Context):
        def create_result(self, argument, data):
            return data

    parsed = Recipe(**recipe)
    builder = InputBuilder(parsed.input, parsed.data_sources)
    group = Groups(parsed.dates, parsed.build.group_by)
    return builder.select(TestContext(parsed), next(iter(group)))


def _times() -> np.ndarray:
    return np.array([np.datetime64(START + datetime.timedelta(hours=FREQUENCY_H * i), "ns") for i in range(N_TIMES)])


def _dates() -> dict:
    last = START + datetime.timedelta(hours=FREQUENCY_H * (N_TIMES - 1))
    return {"start": START.isoformat(), "end": last.isoformat(), "frequency": f"{FREQUENCY_H}h"}


def _field(values: dict, dims: tuple, coords: dict, units: dict) -> xr.Dataset:
    ds = xr.Dataset({name: (dims, data) for name, data in values.items()}, coords=coords)
    for name, unit in units.items():
        ds[name].attrs["units"] = unit
    return ds


@pytest.fixture
def pressure_store(tmp_path) -> str:
    """A pressure-level store whose values differ per time and per level.

    Distinct values everywhere mean a filter that paired the wrong level or the
    wrong time would produce a detectably wrong answer rather than a plausible
    one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    str
        Path to the written Zarr store.
    """
    shape = (N_TIMES, len(LEVELS), N_LAT, N_LON)
    r = np.empty(shape)
    t = np.empty(shape)
    u = np.empty(shape)
    v = np.empty(shape)
    for i in range(N_TIMES):
        for j, level in enumerate(LEVELS):
            r[i, j] = 40.0 + 10 * j + 5 * i  # per cent
            t[i, j] = 250.0 + 10 * j + i
            u[i, j] = 3.0 + j + i
            v[i, j] = 4.0 + j + i

    ds = _field(
        {"r": r, "t": t, "u": u, "v": v},
        ("time", "level", "latitude", "longitude"),
        {
            "time": ("time", _times(), {"standard_name": "time"}),
            "level": ("level", np.array(LEVELS), {"units": "hPa"}),
            "latitude": np.linspace(-60.0, 60.0, N_LAT),
            "longitude": np.linspace(0.0, 270.0, N_LON),
        },
        {"r": "1", "t": "K", "u": "m s**-1", "v": "m s**-1"},
    )

    path = tmp_path / "pressure.zarr"
    ds.to_zarr(path)
    return str(path)


@pytest.fixture
def surface_store(tmp_path) -> str:
    """A single-level store with a 2 m humidity and temperature.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by pytest.

    Returns
    -------
    str
        Path to the written Zarr store.
    """
    shape = (N_TIMES, N_LAT, N_LON)
    r2m = np.full(shape, 80.0)
    t2m = np.full(shape, 290.0)
    for i in range(N_TIMES):
        r2m[i] += 5 * i
        t2m[i] += i

    ds = _field(
        {"r2m": r2m, "2t": t2m},
        ("time", "latitude", "longitude"),
        {
            "time": ("time", _times(), {"standard_name": "time"}),
            "latitude": np.linspace(-60.0, 60.0, N_LAT),
            "longitude": np.linspace(0.0, 270.0, N_LON),
        },
        {"r2m": "1", "2t": "K"},
    )

    path = tmp_path / "surface.zarr"
    ds.to_zarr(path)
    return str(path)


GRID_RULES = {
    "latitude": {"name": "latitude"},
    "longitude": {"name": "longitude"},
    "time": {"name": "time"},
}
PRESSURE_FLAVOUR = {"rules": {**GRID_RULES, "level": {"name": "level"}}, "levtype": "pl"}
SURFACE_FLAVOUR = {"rules": GRID_RULES, "levtype": "sfc"}


def _by_param(fields) -> dict:
    """Group fields by parameter name.

    Parameters
    ----------
    fields : ekd.FieldList
        The fields to group.

    Returns
    -------
    dict
        Parameter name to list of fields.
    """
    out: dict = {}
    for field in fields:
        out.setdefault(field.metadata("param"), []).append(field)
    return out


# ---------------------------------------------------------------------------
# r-to-q
# ---------------------------------------------------------------------------


def test_r_to_q_over_an_xarray_source(pressure_store: str) -> None:
    """Relative humidity and temperature are combined into specific humidity."""
    fields = _run_one_group(
        {
            "dates": _dates(),
            "input": {
                "pipe": [
                    {
                        "xarray-zarr": {
                            "url": pressure_store,
                            "param": ["r", "t"],
                            "flavour": PRESSURE_FLAVOUR,
                        }
                    },
                    {
                        "r-to-q": {
                            "relative_humidity": "r",
                            "temperature": "t",
                            "humidity": "q",
                            "return_inputs": ["temperature"],
                        }
                    },
                ]
            },
        }
    )

    by_param = _by_param(fields)

    # r is consumed, t is kept, q is produced -- on every level and time.
    assert sorted(by_param) == ["q", "t"]
    assert len(by_param["q"]) == N_TIMES * len(LEVELS)
    assert len(by_param["t"]) == N_TIMES * len(LEVELS)


def test_r_to_q_pairs_each_level_with_its_own_temperature(pressure_store: str) -> None:
    """Each level's humidity is computed from that level's own temperature.

    A grouping key that ignored the level would silently combine a humidity
    with a temperature from somewhere else, which is why the store holds a
    different value on every level.
    """
    import earthkit.meteo.thermo.array as thermo

    fields = _run_one_group(
        {
            "dates": _dates(),
            "input": {
                "pipe": [
                    {
                        "xarray-zarr": {
                            "url": pressure_store,
                            "param": ["r", "t"],
                            "flavour": PRESSURE_FLAVOUR,
                        }
                    },
                    {
                        "r-to-q": {
                            "relative_humidity": "r",
                            "temperature": "t",
                            "humidity": "q",
                            "return_inputs": ["temperature"],
                        }
                    },
                ]
            },
        }
    )

    checked = 0
    for field in fields:
        if field.metadata("param") != "q":
            continue
        level = field.metadata("levelist")
        j = LEVELS.index(level)
        i = 0 if field.metadata("valid_datetime") == START.isoformat() else 1

        expected = thermo.specific_humidity_from_relative_humidity(
            np.array([250.0 + 10 * j + i]),
            np.array([40.0 + 10 * j + 5 * i]),
            np.array([100.0 * level]),
        )
        assert field.to_numpy().flatten()[0] == pytest.approx(expected[0]), (level, i)
        checked += 1

    assert checked == N_TIMES * len(LEVELS)


def test_r_to_q_result_is_physical(pressure_store: str) -> None:
    """The humidity comes out in a plausible range, not orders of magnitude off."""
    fields = _run_one_group(
        {
            "dates": _dates(),
            "input": {
                "pipe": [
                    {
                        "xarray-zarr": {
                            "url": pressure_store,
                            "param": ["r", "t"],
                            "flavour": PRESSURE_FLAVOUR,
                        }
                    },
                    {
                        "r-to-q": {
                            "relative_humidity": "r",
                            "temperature": "t",
                            "humidity": "q",
                            "return_inputs": ["temperature"],
                        }
                    },
                ]
            },
        }
    )

    values = np.concatenate([f.to_numpy().flatten() for f in _by_param(fields)["q"]])

    assert np.all(values > 1e-5)
    assert np.all(values < 5e-2)


# ---------------------------------------------------------------------------
# r-to-d
# ---------------------------------------------------------------------------


def test_r_to_d_over_an_xarray_source(surface_store: str) -> None:
    """A 2 m dewpoint is built from the 2 m humidity and temperature."""
    fields = _run_one_group(
        {
            "dates": _dates(),
            "input": {
                "pipe": [
                    {
                        "xarray-zarr": {
                            "url": surface_store,
                            "param": ["r2m", "2t"],
                            "flavour": SURFACE_FLAVOUR,
                        }
                    },
                    {
                        "r-to-d": {
                            "relative_humidity": "r2m",
                            "temperature": "2t",
                            "dewpoint": "2d",
                            "return_inputs": ["temperature"],
                        }
                    },
                ]
            },
        }
    )

    by_param = _by_param(fields)

    assert sorted(by_param) == ["2d", "2t"]
    assert len(by_param["2d"]) == N_TIMES

    # Below saturation the dewpoint is below the temperature, and not by a
    # freakish amount.
    for dewpoint, temperature in zip(by_param["2d"], by_param["2t"]):
        d = dewpoint.to_numpy()
        t = temperature.to_numpy()
        assert np.all(d < t)
        assert np.all(t - d < 15.0)


# ---------------------------------------------------------------------------
# uv-to-ddff
# ---------------------------------------------------------------------------


def test_uv_to_ddff_over_an_xarray_source(pressure_store: str) -> None:
    """Wind components are combined into speed and direction.

    Unlike the humidity filters above, this one survives an empty ``mars``
    namespace: ``u`` and ``v`` share their units, so the fallback grouping key
    does not tell them apart and they still meet. It is covered here because it
    is the third component filter an Xarray recipe is likely to use, and
    because checking the result per level and per time guards against the
    components being paired across levels or times.
    """
    fields = _run_one_group(
        {
            "dates": _dates(),
            "input": {
                "pipe": [
                    {
                        "xarray-zarr": {
                            "url": pressure_store,
                            "param": ["u", "v"],
                            "flavour": PRESSURE_FLAVOUR,
                        }
                    },
                    {
                        "uv-to-ddff": {
                            "u_component": "u",
                            "v_component": "v",
                            "wind_speed": "ff",
                            "wind_direction": "dd",
                            "convention": "meteo",
                        }
                    },
                ]
            },
        }
    )

    by_param = _by_param(fields)

    assert sorted(by_param) == ["dd", "ff"]
    assert len(by_param["ff"]) == N_TIMES * len(LEVELS)

    for field in by_param["ff"]:
        j = LEVELS.index(field.metadata("levelist"))
        i = 0 if field.metadata("valid_datetime") == START.isoformat() else 1
        expected = np.hypot(3.0 + j + i, 4.0 + j + i)
        assert field.to_numpy().flatten()[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The mechanism that broke
# ---------------------------------------------------------------------------


def test_components_group_together_despite_differing_units(pressure_store: str) -> None:
    """Components with different units still belong to the same group.

    This is the failure that made the filters above unusable: with an empty
    ``mars`` namespace the grouping falls back to the whole metadata, ``units``
    included, so a humidity in ``1`` never joins a temperature in ``K``.
    """
    from anemoi.transform.grouping import GroupByParam

    fields = _run_one_group(
        {
            "dates": _dates(),
            "input": {
                "xarray-zarr": {
                    "url": pressure_store,
                    "param": ["r", "t"],
                    "flavour": PRESSURE_FLAVOUR,
                }
            },
        }
    )

    units = {f.metadata("param"): f.metadata("units") for f in fields}
    assert units == {"r": "1", "t": "K"}

    groups = list(GroupByParam(["r", "t"]).iterate(list(fields)))

    # One group per (time, level), each holding both components.
    assert len(groups) == N_TIMES * len(LEVELS)
    for group in groups:
        assert sorted(f.metadata("param") for f in group) == ["r", "t"]
