# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import pytest

from anemoi.datasets import open_dataset


# --------------------------------------------------------------------------
# Value generators
# --------------------------------------------------------------------------
def test_constant_generator() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ConstantValue

    g = ConstantValue(3.5)
    out = g.generate(date_indices=[0, 1], n_ensemble=1, n_grid=4, n_vars=1, var_index=0, seed=0)
    assert out.shape == (2, 1, 4)
    assert (out == 3.5).all()
    assert g.is_constant is True


def test_random_generator_is_deterministic() -> None:
    from anemoi.datasets.usage.gridded.synthetic import RandomValue

    g = RandomValue(mean=10.0, std=2.0)
    kw = dict(date_indices=[0, 1, 2], n_ensemble=2, n_grid=5, n_vars=4, var_index=1, seed=7)
    np.testing.assert_array_equal(g.generate(**kw), g.generate(**kw))

    kw_other = dict(kw, var_index=2)
    assert not np.array_equal(g.generate(**kw), g.generate(**kw_other))
    assert g.is_constant is False

    # mean and std are actually wired through
    assert abs(float(g.generate(**kw).mean()) - 10.0) < 1.0

    # Each date is seeded independently of the range it is drawn in.
    one = g.generate(date_indices=[2], n_ensemble=2, n_grid=5, n_vars=4, var_index=1, seed=7)
    np.testing.assert_array_equal(one[0], g.generate(**kw)[2])


# --------------------------------------------------------------------------
# values spec: one-of type-key dict + scalar / string shorthand
# --------------------------------------------------------------------------
def test_build_value_generator_constant_takes_value_directly() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ConstantValue
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    g = build_value_generator({"constant": 273.15})
    assert isinstance(g, ConstantValue)
    assert g.value == 273.15


def test_build_value_generator_random_with_params() -> None:
    from anemoi.datasets.usage.gridded.synthetic import RandomValue
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    g = build_value_generator({"random": {"mean": 2.0, "std": 3.0}})
    assert isinstance(g, RandomValue)
    assert (g.mean, g.std) == (2.0, 3.0)


def test_build_value_generator_random_defaults_when_payload_empty() -> None:
    from anemoi.datasets.usage.gridded.synthetic import RandomValue
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    g = build_value_generator({"random": {}})
    assert isinstance(g, RandomValue)
    assert (g.mean, g.std) == (0.0, 1.0)


def test_build_value_generator_bare_scalar_is_constant() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ConstantValue
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    g = build_value_generator(5.0)
    assert isinstance(g, ConstantValue)
    assert g.value == 5.0


def test_build_value_generator_bare_string_is_named_generator_with_defaults() -> None:
    from anemoi.datasets.usage.gridded.synthetic import RandomValue
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    g = build_value_generator("random")
    assert isinstance(g, RandomValue)
    assert (g.mean, g.std) == (0.0, 1.0)


def test_build_value_generator_rejects_bool() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="must be"):
        build_value_generator(True)


def test_build_value_generator_rejects_multi_key_dict() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="exactly one"):
        build_value_generator({"constant": 1.0, "random": {}})


def test_build_value_generator_rejects_unknown_type() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="constant"):
        build_value_generator({"sinewave": {}})


def test_build_value_generator_bare_string_constant_needs_value() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="constant"):
        build_value_generator("constant")


def test_build_value_generator_constant_rejects_dict_payload() -> None:
    # constant takes its value directly; a nested {"value": ...} is not the v2 form.
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="number"):
        build_value_generator({"constant": {"value": 1.0}})


# --------------------------------------------------------------------------
# geography (renamed from grid; keeps the four resolvers)
# --------------------------------------------------------------------------
def test_resolve_bbox_geography() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_geography

    lat, lon, field_shape = resolve_geography({"bbox": [10, 0, 0, 10], "resolution": 1.0})
    assert field_shape == (11, 11)
    assert lat.shape == (121,)
    assert lat[0] == 10.0 and lat[-1] == 0.0
    assert lon[0] == 0.0 and lon[-1] == 10.0
    assert (lat[10], lon[10]) == (10.0, 10.0)


def test_resolve_bbox_requires_resolution() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_geography

    with pytest.raises(ValueError, match="requires a 'resolution'"):
        resolve_geography({"bbox": [10, 0, 0, 10]})


def test_resolve_bbox_rejects_inverted_bounds() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_geography

    with pytest.raises(ValueError, match="south must be <= north"):
        resolve_geography({"bbox": [0, 0, 10, 10], "resolution": 1.0})
    with pytest.raises(ValueError, match="east must be >= west"):
        resolve_geography({"bbox": [10, 10, 0, 0], "resolution": 1.0})


def test_resolve_unstructured_from_arrays() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_geography

    lat, lon, field_shape = resolve_geography(
        {"unstructured": {"latitudes": [1.0, 2.0, 3.0], "longitudes": [4.0, 5.0, 6.0]}}
    )
    assert field_shape == (3,)
    np.testing.assert_array_equal(lat, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(lon, [4.0, 5.0, 6.0])


def test_resolve_geography_rejects_unknown_type() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_geography

    with pytest.raises(ValueError, match="exactly one of"):
        resolve_geography({"hexagon": 1})


def test_resolve_named_geography(monkeypatch) -> None:
    from anemoi.datasets.usage.gridded import synthetic

    fake = {"latitudes": np.array([1.0, 2.0]), "longitudes": np.array([3.0, 4.0])}
    monkeypatch.setattr("anemoi.transform.grids.named.lookup", lambda name: fake)
    lat, lon, field_shape = synthetic.resolve_geography({"named": "o96"})
    assert field_shape == (2,)
    np.testing.assert_array_equal(lat, [1.0, 2.0])


def test_resolve_icon_geography(monkeypatch) -> None:
    from anemoi.datasets.usage.gridded import synthetic

    class FakeIconGrid:
        def __init__(self, path, refinement_level_c=None):
            self.path = path

        def latlon(self):
            return np.array([5.0, 6.0]), np.array([7.0, 8.0])

    monkeypatch.setattr("anemoi.transform.grids.icon.IconGrid", FakeIconGrid)
    lat, lon, field_shape = synthetic.resolve_geography({"icon": {"path": "/fake/grid.nc"}})
    assert field_shape == (2,)
    np.testing.assert_array_equal(lon, [7.0, 8.0])


# --------------------------------------------------------------------------
# config parsing (v2 surface)
# --------------------------------------------------------------------------
def _minimal_raw(**overrides):
    # start/end/frequency may be passed flat for convenience; they are assembled
    # into the nested 'dates' block the config actually expects.
    dates = {"start": "2020-01-01", "end": "2020-01-02", "frequency": "6h"}
    for key in ("start", "end", "frequency"):
        if key in overrides:
            dates[key] = overrides.pop(key)
    raw = {
        "geography": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
        "variables": ["a", "b"],
        "dates": dates,
        "layout": "gridded",
    }
    raw.update(overrides)
    return raw


def test_parse_config_basic() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw())
    assert cfg.variables == ["a", "b"]
    assert len(cfg.dates) == 5  # 00, 06, 12, 18, 00
    assert cfg.frequency == datetime.timedelta(hours=6)
    assert cfg.n_ensemble == 1
    assert cfg.latitudes.size == 9  # 3x3 grid
    assert len(cfg.generators) == 2


def test_parse_config_rejects_integer_variables() -> None:
    # the int-count form is dropped in v2; variables must be a list.
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="list"):
        parse_synthetic_config(_minimal_raw(variables=12))


def test_parse_config_variable_dict_entries() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ConstantValue
    from anemoi.datasets.usage.gridded.synthetic import RandomValue
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(
        _minimal_raw(
            variables=[
                {"name": "a", "values": {"constant": 9.0}},
                "b",  # string -> default generator
            ]
        )
    )
    assert cfg.variables == ["a", "b"]
    assert isinstance(cfg.generators[0], ConstantValue)
    assert isinstance(cfg.generators[1], RandomValue)  # default


def test_parse_config_top_level_values_is_default() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ConstantValue
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(values={"constant": 1.0}))
    assert all(isinstance(g, ConstantValue) for g in cfg.generators)


def test_parse_config_per_variable_overrides_default() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ConstantValue
    from anemoi.datasets.usage.gridded.synthetic import RandomValue
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(
        _minimal_raw(
            variables=[{"name": "a", "values": "random"}, "b"],
            values={"constant": 1.0},
        )
    )
    assert isinstance(cfg.generators[0], RandomValue)  # per-variable wins
    assert isinstance(cfg.generators[1], ConstantValue)  # default


def test_parse_config_variable_dict_requires_name() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="name"):
        parse_synthetic_config(_minimal_raw(variables=[{"values": {"constant": 1.0}}]))


def test_parse_config_rejects_duplicate_variable_names() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="duplicate"):
        parse_synthetic_config(_minimal_raw(variables=["a", "a"]))


def test_parse_config_threads_variable_metadata() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(variables=[{"name": "2t", "metadata": {"mars": {"param": "T_2M"}}}, "b"]))
    assert cfg.variables_metadata["2t"] == {"mars": {"param": "T_2M"}}
    assert cfg.variables_metadata["b"] == {}


def test_parse_config_per_variable_statistics_override() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(
        _minimal_raw(variables=[{"name": "a", "values": {"constant": 1.0}, "statistics": {"mean": 99.0}}, "b"])
    )
    assert cfg.stats_overrides[0] == {"mean": 99.0}
    assert cfg.stats_overrides[1] is None


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
def test_parse_config_requires_layout() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    raw = _minimal_raw()
    del raw["layout"]
    with pytest.raises(ValueError, match="layout"):
        parse_synthetic_config(raw)


def test_parse_config_rejects_unimplemented_layout() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(NotImplementedError, match="tabular"):
        parse_synthetic_config(_minimal_raw(layout="tabular"))
    with pytest.raises(NotImplementedError, match="trajectories"):
        parse_synthetic_config(_minimal_raw(layout="trajectories"))


def test_parse_config_rejects_unknown_layout() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="layout"):
        parse_synthetic_config(_minimal_raw(layout="nonsense"))


# --------------------------------------------------------------------------
# ensembles (renamed from ensemble)
# --------------------------------------------------------------------------
def test_parse_config_ensembles() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(ensembles=4))
    assert cfg.n_ensemble == 4


def test_parse_config_rejects_old_ensemble_key() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="unknown synthetic keys"):
        parse_synthetic_config(_minimal_raw(ensemble=4))


def test_parse_config_rejects_non_positive_ensembles() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="ensembles' must be a positive integer"):
        parse_synthetic_config(_minimal_raw(ensembles=0))


# --------------------------------------------------------------------------
# dates / misc parsing (carried from v1)
# --------------------------------------------------------------------------
def test_parse_config_rejects_unknown_key() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="unknown synthetic keys"):
        parse_synthetic_config(_minimal_raw(colour="blue"))


def test_parse_config_rejects_missing_required_key() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    raw = _minimal_raw()
    del raw["dates"]
    with pytest.raises(ValueError, match="missing required key 'dates'"):
        parse_synthetic_config(raw)


def test_parse_config_rejects_end_before_start() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="must not be before"):
        parse_synthetic_config(_minimal_raw(start="2020-01-05", end="2020-01-01"))


def test_parse_config_dtype_seed_overrides() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(dtype="float64", seed=123))
    assert cfg.dtype == np.dtype("float64")
    assert cfg.seed == 123


def test_parse_config_rejects_integer_dtype_with_random() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="'random'"):
        parse_synthetic_config(_minimal_raw(dtype="int32", values="random"))


def test_parse_config_rejects_integer_dtype_with_fractional_constant() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="is not an integer"):
        parse_synthetic_config(_minimal_raw(dtype="int32", values={"constant": 3.5}))


# --------------------------------------------------------------------------
# computed forcings
# --------------------------------------------------------------------------
def test_computed_forcing_generator_matches_earthkit() -> None:
    from earthkit.data import from_source

    from anemoi.datasets.usage.gridded.synthetic import ComputedForcingValue

    lat = np.array([0.0, 30.0, -45.0, 60.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    dates = np.array(["2020-01-01T00", "2020-06-01T12"], dtype="datetime64[s]")

    g = ComputedForcingValue("insolation", latitudes=lat, longitudes=lon, dates=dates)
    out = g.generate(date_indices=[0, 1], n_ensemble=1, n_grid=4, n_vars=1, var_index=0, seed=0)
    assert out.shape == (2, 1, 4)

    fl = from_source(
        "forcings",
        latitudes=lat,
        longitudes=lon,
        date=[d.astype("datetime64[s]").astype(datetime.datetime) for d in dates],
        param=["insolation"],
    )
    expected = np.stack([f.to_numpy(flatten=True) for f in fl])  # (2, 4)
    np.testing.assert_allclose(out[:, 0, :], expected)


def test_computed_forcing_broadcasts_over_ensemble() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ComputedForcingValue

    lat = np.array([0.0, 30.0])
    lon = np.array([0.0, 90.0])
    dates = np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[s]")
    g = ComputedForcingValue("cos_latitude", latitudes=lat, longitudes=lon, dates=dates)
    out = g.generate(date_indices=[0, 1], n_ensemble=3, n_grid=2, n_vars=1, var_index=0, seed=0)
    assert out.shape == (2, 3, 2)
    # all ensemble members identical
    np.testing.assert_array_equal(out[:, 0, :], out[:, 1, :])


def test_computed_forcing_constant_in_time_is_marked_constant() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ComputedForcingValue

    lat = np.array([0.0, 30.0])
    lon = np.array([0.0, 90.0])
    dates = np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[s]")
    assert ComputedForcingValue("cos_latitude", latitudes=lat, longitudes=lon, dates=dates).is_constant is True
    assert ComputedForcingValue("insolation", latitudes=lat, longitudes=lon, dates=dates).is_constant is False


def test_parse_config_routes_forcing_string_to_computed_generator() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ComputedForcingValue
    from anemoi.datasets.usage.gridded.synthetic import RandomValue
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(variables=["2t", "insolation", "cos_latitude"]))
    assert isinstance(cfg.generators[0], RandomValue)  # ordinary variable
    assert isinstance(cfg.generators[1], ComputedForcingValue)
    assert isinstance(cfg.generators[2], ComputedForcingValue)


def test_parse_config_forcing_with_values_block_is_error() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="forcing"):
        parse_synthetic_config(_minimal_raw(variables=[{"name": "insolation", "values": {"constant": 1.0}}, "b"]))


def test_parse_config_forcing_metadata_marks_computed_forcing() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(variables=["cos_latitude", "insolation"]))
    assert cfg.variables_metadata["cos_latitude"]["computed_forcing"] is True
    assert cfg.variables_metadata["cos_latitude"]["constant_in_time"] is True
    assert cfg.variables_metadata["insolation"]["constant_in_time"] is False


# --------------------------------------------------------------------------
# dataset behaviour
# --------------------------------------------------------------------------
def _dataset(**overrides):
    from anemoi.datasets.usage.gridded.synthetic import SyntheticGriddedDataset
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    return SyntheticGriddedDataset(parse_synthetic_config(_minimal_raw(**overrides)))


def test_dataset_is_a_gridded_zarr() -> None:
    from anemoi.datasets.usage.gridded.store import GriddedZarr

    assert isinstance(_dataset(), GriddedZarr)


def test_synthetic_array_generates_only_requested_dates(monkeypatch) -> None:
    from anemoi.datasets.usage.gridded import synthetic

    calls = []
    real_generate = synthetic._SyntheticArray._generate

    def spy(self, date_indices):
        calls.append(np.asarray(date_indices).tolist())
        return real_generate(self, date_indices)

    monkeypatch.setattr(synthetic._SyntheticArray, "_generate", spy)

    ds = _dataset(start="2000-01-01", end="2030-01-01", frequency="6h")
    assert len(ds) > 40_000
    assert calls == []  # opening generated nothing

    ds[7]
    assert calls == [[7]]

    ds[10:13]
    assert calls[-1] == [10, 11, 12]


def test_dataset_shape_and_descriptors() -> None:
    ds = _dataset(variables=["a", "b", "c"])
    assert ds.shape == (5, 3, 1, 9)
    assert len(ds) == 5
    assert ds.variables == ["a", "b", "c"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2}
    assert ds.field_shape == (3, 3)
    assert ds.latitudes.shape == (9,)
    assert ds.missing == set()
    assert ds.dtype == np.dtype("float32")


def test_dataset_getitem_constant() -> None:
    ds = _dataset(variables=[{"name": "x", "values": {"constant": 5.0}}])
    np.testing.assert_array_equal(ds[0], np.full((1, 1, 9), 5.0, dtype=np.float32))


def test_dataset_getitem_rejects_out_of_bounds_index() -> None:
    ds = _dataset()  # 5 dates
    with pytest.raises(IndexError):
        ds[99]
    with pytest.raises(IndexError):
        ds[-99]
    with pytest.raises(IndexError):
        ds[[0, 99]]


def test_dataset_getitem_boolean_mask_selects_true_positions() -> None:
    ds = _dataset(variables=[{"name": "x", "values": {"constant": 2.0}}])  # 5 dates
    mask = np.array([True, False, True, False, True])
    np.testing.assert_array_equal(ds[mask], ds[[0, 2, 4]])


def test_dataset_getitem_numpy_array_on_variable_axis() -> None:
    ds = _dataset(variables=["a", "b", "c"])  # shape (5, 3, 1, 9)
    selected = ds[:, np.array([0, 2])]
    assert selected.shape == (5, 2, 1, 9)


def test_dataset_statistics_and_constant_fields() -> None:
    ds = _dataset(
        variables=[
            {"name": "k", "values": {"constant": 7.0}},
            {"name": "r", "values": {"random": {"mean": 0.0, "std": 1.0}}},
        ],
    )
    np.testing.assert_array_equal(ds.statistics["mean"][0], 7.0)
    np.testing.assert_array_equal(ds.statistics["stdev"][0], 0.0)
    assert ds.constant_fields == ["k"]


def test_dataset_statistics_respects_override() -> None:
    ds = _dataset(
        variables=[
            {"name": "k", "values": {"constant": 7.0}, "statistics": {"mean": 99.0}},
            "b",
        ],
    )
    assert ds.statistics["mean"][0] == 99.0
    # unoverridden keys keep the analytic value
    assert ds.statistics["stdev"][0] == 0.0


def test_dataset_tendency_statistics_for_constant_is_zero() -> None:
    ds = _dataset(variables=[{"name": "k", "values": {"constant": 7.0}}])
    tend = ds.statistics_tendencies()
    np.testing.assert_array_equal(tend["mean"], [0.0])
    np.testing.assert_array_equal(tend["stdev"], [0.0])


def test_dataset_constant_fields_single_date_is_analytic() -> None:
    ds = _dataset(
        start="2020-01-01",
        end="2020-01-01",
        variables=[
            {"name": "k", "values": {"constant": 7.0}},
            {"name": "r", "values": "random"},
        ],
    )
    assert len(ds) == 1
    assert ds.constant_fields == ["k"]


def test_dataset_metadata_roundtrip() -> None:
    meta = _dataset(variables=["a", "b"]).metadata()
    assert meta["variables"] == ["a", "b"]
    assert meta["specific"]["synthetic"] is True
    assert meta["shape"][1] == 2


def test_dataset_variables_metadata_surfaced() -> None:
    ds = _dataset(variables=[{"name": "2t", "metadata": {"mars": {"param": "T_2M"}}}, "b"])
    assert ds.metadata()["variables_metadata"]["2t"] == {"mars": {"param": "T_2M"}}


# --------------------------------------------------------------------------
# end to end via open_dataset
# --------------------------------------------------------------------------
def test_open_dataset_synthetic_bbox_end_to_end() -> None:
    ds = open_dataset(
        synthetic={
            "geography": {"bbox": [10, 0, 0, 10], "resolution": 1.0},
            "variables": ["a", "b"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "6h"},
            "layout": "gridded",
            "values": {"constant": 5.0},
        }
    )
    assert ds.shape == (5, 2, 1, 121)
    assert ds.field_shape == (11, 11)
    np.testing.assert_array_equal(ds[0], np.full((2, 1, 121), 5.0, dtype=np.float32))


def test_open_dataset_synthetic_random_is_reproducible() -> None:
    spec = dict(
        geography={"bbox": [4, 0, 0, 4], "resolution": 2.0},
        variables=["a", "b", "c"],
        dates={"start": "2020-01-01", "end": "2020-01-01", "frequency": "6h"},
        layout="gridded",
        values={"random": {"mean": 0.0, "std": 1.0}},
        seed=42,
    )
    first = open_dataset(synthetic=dict(spec))
    second = open_dataset(synthetic=dict(spec))
    np.testing.assert_array_equal(first[:], second[:])


def test_open_dataset_synthetic_with_forcing_end_to_end() -> None:
    ds = open_dataset(
        synthetic={
            "geography": {"bbox": [10, 0, 0, 10], "resolution": 2.0},
            "variables": ["2t", "insolation", "cos_latitude"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "6h"},
            "layout": "gridded",
        }
    )
    assert ds.variables == ["2t", "insolation", "cos_latitude"]
    block = ds[0]  # (3, 1, 36)
    assert block.shape == (3, 1, 36)
    # cos_latitude is in [-1, 1] and constant in time
    cos_lat = ds.name_to_index["cos_latitude"]
    assert np.all(np.abs(ds[0][cos_lat]) <= 1.0 + 1e-9)
    np.testing.assert_allclose(ds[0][cos_lat], ds[3][cos_lat])


def test_open_dataset_synthetic_composes_with_select() -> None:
    ds = open_dataset(
        synthetic={
            "geography": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
            "variables": ["a", "b", "c"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "1d"},
            "layout": "gridded",
        },
        select=["a", "c"],
    )
    assert ds.variables == ["a", "c"]
    assert ds.shape[1] == 2


def test_open_dataset_synthetic_composes_with_date_subsetting() -> None:
    spec = {
        "geography": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
        "variables": ["a", "b"],
        "dates": {"start": "2020-01-01", "end": "2020-01-05", "frequency": "1d"},
        "layout": "gridded",
    }
    assert len(open_dataset(synthetic=dict(spec))) == 5
    subset = open_dataset(synthetic=dict(spec), start="2020-01-02", end="2020-01-04")
    assert len(subset) == 3


def test_open_dataset_synthetic_composes_with_rename() -> None:
    ds = open_dataset(
        synthetic={
            "geography": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
            "variables": ["a", "b"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "1d"},
            "layout": "gridded",
        },
        rename={"a": "temperature"},
    )
    assert "temperature" in ds.variables
    assert "a" not in ds.variables


def test_open_dataset_synthetic_ensembles() -> None:
    ds = open_dataset(
        synthetic={
            "geography": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
            "variables": ["a", "b"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "1d"},
            "layout": "gridded",
            "ensembles": 5,
        }
    )
    assert ds.shape == (2, 2, 5, 9)
