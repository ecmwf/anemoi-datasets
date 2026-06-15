# (C) Copyright 2025 Anemoi contributors.
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


def test_index_generator_encodes_position() -> None:
    from anemoi.datasets.usage.gridded.synthetic import IndexEncodedValue

    g = IndexEncodedValue()
    out = g.generate(date_indices=[0, 1], n_ensemble=1, n_grid=3, n_vars=2, var_index=1, seed=0)
    # value = ((date * n_vars + var) * n_ensemble + ens) * n_grid + grid
    assert out[0, 0, 0] == ((0 * 2 + 1) * 1 + 0) * 3 + 0
    assert out[1, 0, 2] == ((1 * 2 + 1) * 1 + 0) * 3 + 2
    assert g.is_constant is False


def test_index_generator_statistics_match_brute_force() -> None:
    from anemoi.datasets.usage.gridded.synthetic import IndexEncodedValue

    g = IndexEncodedValue()
    kw = dict(n_dates=4, n_ensemble=2, n_grid=3, n_vars=5, var_index=2, seed=0)
    full = g.generate(date_indices=np.arange(4), n_ensemble=2, n_grid=3, n_vars=5, var_index=2, seed=0)

    s = g.statistics(**kw)
    assert s["mean"] == pytest.approx(float(full.mean()))
    assert s["stdev"] == pytest.approx(float(full.std()))
    assert s["maximum"] == pytest.approx(float(full.max()))
    assert s["minimum"] == pytest.approx(float(full.min()))

    t = g.tendency_statistics(**kw)
    diff = np.diff(full, axis=0)
    assert t["mean"] == pytest.approx(float(diff.mean()))
    assert t["stdev"] == pytest.approx(float(diff.std()))


def test_build_value_generator_rejects_unknown_mode() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="Unknown synthetic value mode"):
        build_value_generator({"mode": "sinewave"})


def test_build_value_generator_rejects_non_dict() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="must be a dict"):
        build_value_generator("constant")


def test_build_value_generator_rejects_missing_mode() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="missing required key 'mode'"):
        build_value_generator({})


def test_build_value_generator_constant_requires_value() -> None:
    from anemoi.datasets.usage.gridded.synthetic import build_value_generator

    with pytest.raises(ValueError, match="requires a 'value'"):
        build_value_generator({"mode": "constant"})


def test_resolve_bbox_grid() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_grid

    lat, lon, field_shape = resolve_grid({"bbox": [10, 0, 0, 10], "resolution": 1.0})
    assert field_shape == (11, 11)
    assert lat.shape == (121,)
    assert lon.shape == (121,)
    assert lat[0] == 10.0 and lat[-1] == 0.0
    assert lon[0] == 0.0 and lon[-1] == 10.0
    assert (lat[10], lon[10]) == (10.0, 10.0)  # NE corner: lat/lon paired per gridpoint


def test_resolve_bbox_grid_coordinates_are_exact() -> None:
    # A float resolution must not leave floating-point fuzz on the grid edges.
    from anemoi.datasets.usage.gridded.synthetic import resolve_grid

    lat, lon, field_shape = resolve_grid({"bbox": [10, 0, 0, 10], "resolution": 0.1})
    assert field_shape == (101, 101)
    assert lat[0] == 10.0
    assert lat[-1] == 0.0
    assert lon[0] == 0.0
    assert lon[-1] == 10.0


def test_resolve_bbox_requires_resolution() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_grid

    with pytest.raises(ValueError, match="requires a 'resolution'"):
        resolve_grid({"bbox": [10, 0, 0, 10]})


def test_resolve_bbox_rejects_inverted_bounds() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_grid

    with pytest.raises(ValueError, match="south must be <= north"):
        resolve_grid({"bbox": [0, 0, 10, 10], "resolution": 1.0})
    with pytest.raises(ValueError, match="east must be >= west"):
        resolve_grid({"bbox": [10, 10, 0, 0], "resolution": 1.0})


def test_latlon_from_npz_accepts_aliases_and_rejects_missing() -> None:
    from anemoi.datasets.usage.gridded.synthetic import _latlon_from_npz

    lat, lon = _latlon_from_npz({"Lat": [1.0, 2.0], "LON": [3.0, 4.0]})
    np.testing.assert_array_equal(lat, [1.0, 2.0])
    np.testing.assert_array_equal(lon, [3.0, 4.0])

    with pytest.raises(ValueError, match="no recognised"):
        _latlon_from_npz({"x": [1.0], "y": [2.0]})


def test_resolve_unstructured_from_arrays() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_grid

    lat, lon, field_shape = resolve_grid(
        {"unstructured": {"latitudes": [1.0, 2.0, 3.0], "longitudes": [4.0, 5.0, 6.0]}}
    )
    assert field_shape == (3,)
    np.testing.assert_array_equal(lat, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(lon, [4.0, 5.0, 6.0])


def test_resolve_grid_rejects_unknown_type() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_grid

    with pytest.raises(ValueError, match="exactly one of"):
        resolve_grid({"hexagon": 1})


def test_resolve_named_grid(monkeypatch) -> None:
    from anemoi.datasets.usage.gridded import synthetic

    fake = {"latitudes": np.array([1.0, 2.0]), "longitudes": np.array([3.0, 4.0])}
    monkeypatch.setattr("anemoi.transform.grids.named.lookup", lambda name: fake)
    lat, lon, field_shape = synthetic.resolve_grid({"named": "o96"})
    assert field_shape == (2,)
    np.testing.assert_array_equal(lat, [1.0, 2.0])


def test_resolve_icon_grid(monkeypatch) -> None:
    from anemoi.datasets.usage.gridded import synthetic

    class FakeIconGrid:
        def __init__(self, path, refinement_level_c=None):
            self.path = path

        def latlon(self):
            return np.array([5.0, 6.0]), np.array([7.0, 8.0])

    monkeypatch.setattr("anemoi.transform.grids.icon.IconGrid", FakeIconGrid)
    lat, lon, field_shape = synthetic.resolve_grid({"icon": {"path": "/fake/grid.nc"}})
    assert field_shape == (2,)
    np.testing.assert_array_equal(lon, [7.0, 8.0])


def test_resolve_icon_requires_path() -> None:
    from anemoi.datasets.usage.gridded.synthetic import resolve_grid

    with pytest.raises(ValueError, match="requires a 'path'"):
        resolve_grid({"icon": {}})


def _minimal_raw(**overrides):
    # start/end/frequency may be passed flat for convenience; they are assembled
    # into the nested 'dates' block the config actually expects.
    dates = {"start": "2020-01-01", "end": "2020-01-02", "frequency": "6h"}
    for key in ("start", "end", "frequency"):
        if key in overrides:
            dates[key] = overrides.pop(key)
    raw = {
        "grid": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
        "variables": ["a", "b"],
        "dates": dates,
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


def test_parse_config_variable_count_autonames() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(variables=12))
    assert cfg.variables[0] == "var_00"
    assert cfg.variables[-1] == "var_11"


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


def test_parse_config_dates_block() -> None:
    # start/end/frequency live under a 'dates' block, mirroring the recipe API.
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(
        {
            "grid": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
            "variables": ["a", "b"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "6h"},
        }
    )
    assert len(cfg.dates) == 5
    assert cfg.frequency == datetime.timedelta(hours=6)


def test_parse_config_rejects_non_dict_dates() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="'dates' must be a dict"):
        parse_synthetic_config(_minimal_raw(dates="2020-01-01"))


def test_parse_config_rejects_dates_missing_subkey() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="'dates' is missing required key 'frequency'"):
        parse_synthetic_config(_minimal_raw(dates={"start": "2020-01-01", "end": "2020-01-02"}))


def test_parse_config_rejects_unknown_dates_subkey() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="unknown synthetic 'dates' keys"):
        parse_synthetic_config(
            _minimal_raw(dates={"start": "2020-01-01", "end": "2020-01-02", "frequency": "6h", "group_by": "monthly"})
        )


def test_parse_config_rejects_end_before_start() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="must not be before"):
        parse_synthetic_config(_minimal_raw(start="2020-01-05", end="2020-01-01"))


def test_parse_config_per_variable_values() -> None:
    from anemoi.datasets.usage.gridded.synthetic import ConstantValue
    from anemoi.datasets.usage.gridded.synthetic import RandomValue
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(
        _minimal_raw(
            values={
                "default": {"mode": "random", "mean": 0.0, "std": 1.0},
                "a": {"mode": "constant", "value": 9.0},
            }
        )
    )
    assert isinstance(cfg.generators[0], ConstantValue)  # "a"
    assert isinstance(cfg.generators[1], RandomValue)  # "b" -> default


def test_parse_config_single_date_when_start_equals_end() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(start="2020-01-01", end="2020-01-01"))
    assert len(cfg.dates) == 1


def test_parse_config_does_not_overshoot_unaligned_end() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    # 00:00 -> 07:00 at 6h: dates are 00:00 and 06:00, never past 07:00
    cfg = parse_synthetic_config(_minimal_raw(start="2020-01-01T00:00", end="2020-01-01T07:00"))
    assert len(cfg.dates) == 2
    assert cfg.dates[-1] == np.datetime64("2020-01-01T06:00:00")


def test_parse_config_dtype_seed_ensemble_overrides() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(_minimal_raw(dtype="float64", seed=123, ensemble=4))
    assert cfg.dtype == np.dtype("float64")
    assert cfg.seed == 123
    assert cfg.n_ensemble == 4


def test_parse_config_rejects_non_positive_ensemble() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="ensemble' must be a positive integer"):
        parse_synthetic_config(_minimal_raw(ensemble=0))


def test_parse_config_rejects_non_dict_values() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="'values' must be a dict"):
        parse_synthetic_config(_minimal_raw(values=[]))


def test_parse_config_rejects_index_mode_with_too_narrow_dtype() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    # ~877k hourly dates x 3 vars x 9 gridpoints overflows float32's exact-integer range.
    with pytest.raises(ValueError, match="'index' value mode"):
        parse_synthetic_config(
            _minimal_raw(
                start="2000-01-01",
                end="2100-01-01",
                frequency="1h",
                variables=3,
                values={"default": {"mode": "index"}},
                dtype="float32",
            )
        )


def test_parse_config_rejects_integer_dtype_with_random_mode() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    # A continuous Gaussian truncated to integers no longer matches its statistics.
    with pytest.raises(ValueError, match="'random' value mode"):
        parse_synthetic_config(_minimal_raw(dtype="int32", values={"default": {"mode": "random"}}))


def test_parse_config_rejects_integer_dtype_with_fractional_constant() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    with pytest.raises(ValueError, match="is not an integer"):
        parse_synthetic_config(_minimal_raw(dtype="int32", values={"default": {"mode": "constant", "value": 3.5}}))


def test_parse_config_allows_integer_dtype_with_integral_values() -> None:
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    cfg = parse_synthetic_config(
        _minimal_raw(
            dtype="int64",
            values={"default": {"mode": "index"}, "a": {"mode": "constant", "value": 7}},
        )
    )
    assert cfg.dtype == np.dtype("int64")


def _dataset(**overrides):
    from anemoi.datasets.usage.gridded.synthetic import SyntheticGriddedDataset
    from anemoi.datasets.usage.gridded.synthetic import parse_synthetic_config

    return SyntheticGriddedDataset(parse_synthetic_config(_minimal_raw(**overrides)))


def test_dataset_is_a_gridded_zarr() -> None:
    # The synthetic dataset inherits the whole Dataset contract from GriddedZarr
    # rather than re-implementing it; this guards that inheritance.
    from anemoi.datasets.usage.gridded.store import GriddedZarr

    assert isinstance(_dataset(), GriddedZarr)


def test_synthetic_array_generates_only_requested_dates(monkeypatch) -> None:
    # Opening a dataset must materialise nothing, and indexing must generate
    # only the dates asked for -- never the whole (possibly enormous) range.
    from anemoi.datasets.usage.gridded import synthetic

    calls = []
    real_generate = synthetic._SyntheticArray._generate

    def spy(self, date_indices):
        calls.append(np.asarray(date_indices).tolist())
        return real_generate(self, date_indices)

    monkeypatch.setattr(synthetic._SyntheticArray, "_generate", spy)

    # A 30-year, 6-hourly range: materialising it would be billions of values.
    ds = _dataset(start="2000-01-01", end="2030-01-01", frequency="6h")
    assert len(ds) > 40_000
    assert calls == []  # opening generated nothing

    ds[7]
    assert calls == [[7]]  # one timestep only

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
    ds = _dataset(variables=["x"], values={"x": {"mode": "constant", "value": 5.0}})
    np.testing.assert_array_equal(ds[0], np.full((1, 1, 9), 5.0, dtype=np.float32))


def test_dataset_getitem_rejects_out_of_bounds_index() -> None:
    # A real GriddedZarr raises on an out-of-range index; the synthetic dataset
    # must not silently fabricate a nonexistent timestep.
    ds = _dataset()  # 5 dates
    with pytest.raises(IndexError):
        ds[99]
    with pytest.raises(IndexError):
        ds[-99]
    with pytest.raises(IndexError):
        ds[[0, 99]]


def test_dataset_getitem_negative_index_wraps() -> None:
    ds = _dataset(variables=["x"], values={"x": {"mode": "index"}})  # 5 dates
    np.testing.assert_array_equal(ds[-1], ds[len(ds) - 1])
    np.testing.assert_array_equal(ds[-5], ds[0])


def test_dataset_getitem_boolean_mask_selects_true_positions() -> None:
    # A boolean mask must select the True positions, not be cast to integers.
    ds = _dataset(variables=["x"], values={"x": {"mode": "index"}})  # 5 dates
    mask = np.array([True, False, True, False, True])
    np.testing.assert_array_equal(ds[mask], ds[[0, 2, 4]])


def test_dataset_getitem_numpy_array_on_variable_axis() -> None:
    # A numpy fancy index on a non-date axis must not crash index expansion.
    ds = _dataset(variables=["a", "b", "c"])  # shape (5, 3, 1, 9)
    selected = ds[:, np.array([0, 2])]
    assert selected.shape == (5, 2, 1, 9)


def test_dataset_statistics_and_constant_fields() -> None:
    ds = _dataset(
        variables=["k", "r"],
        values={
            "k": {"mode": "constant", "value": 7.0},
            "r": {"mode": "random", "mean": 0.0, "std": 1.0},
        },
    )
    np.testing.assert_array_equal(ds.statistics["mean"][0], 7.0)
    np.testing.assert_array_equal(ds.statistics["stdev"][0], 0.0)
    assert ds.constant_fields == ["k"]


def test_dataset_tendency_statistics_for_constant_is_zero() -> None:
    ds = _dataset(variables=["k"], values={"k": {"mode": "constant", "value": 7.0}})
    tend = ds.statistics_tendencies()
    np.testing.assert_array_equal(tend["mean"], [0.0])
    np.testing.assert_array_equal(tend["stdev"], [0.0])


def test_dataset_usage_factory_load_resolves() -> None:
    # A synthetic dataset is a genuine GriddedZarr, so it resolves usage
    # factories like any other dataset.
    ds = _dataset()
    assert ds.usage_factory_load("Subset").__name__ == "Subset"


def test_dataset_tendency_statistics_missing_delta_raises() -> None:
    # Only the dataset-frequency tendency is precomputed; like a real dataset,
    # requesting an unavailable delta raises KeyError.
    ds = _dataset()
    with pytest.raises(KeyError):
        ds.statistics_tendencies(datetime.timedelta(days=30))


def test_dataset_computed_constant_fields_single_date() -> None:
    # A single-date dataset cannot produce tendencies; the inherited
    # computed_constant_fields() must fall back to sample-based detection
    # rather than raising.
    ds = _dataset(start="2020-01-01", end="2020-01-01")
    assert len(ds) == 1
    assert ds.computed_constant_fields() == ["a", "b"]


def test_dataset_constant_fields_single_date_is_analytic() -> None:
    # With one date the inherited sample-based detection cannot tell a constant
    # field from a varying one and marks every field constant; the synthetic
    # dataset must instead report the answer it knows analytically.
    ds = _dataset(
        start="2020-01-01",
        end="2020-01-01",
        variables=["k", "r"],
        values={"k": {"mode": "constant", "value": 7.0}, "r": {"mode": "random"}},
    )
    assert len(ds) == 1
    assert ds.constant_fields == ["k"]


def test_dataset_metadata_roundtrip() -> None:
    # metadata() serializes via json.dumps/json.loads internally; this checks
    # the synthetic dataset round-trips into a checkpoint-style metadata blob.
    meta = _dataset(variables=["a", "b"]).metadata()
    assert meta["variables"] == ["a", "b"]
    assert meta["specific"]["synthetic"] is True
    assert meta["shape"][1] == 2


def test_dataset_metadata_specific() -> None:
    meta = _dataset(variables=["a", "b"]).metadata_specific()
    assert meta["synthetic"] is True
    # the base-class implementation populates these
    assert "action" in meta
    assert meta["variables"] == ["a", "b"]


def test_open_dataset_synthetic_bbox_end_to_end() -> None:
    ds = open_dataset(
        synthetic={
            "grid": {"bbox": [10, 0, 0, 10], "resolution": 1.0},
            "variables": ["a", "b"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "6h"},
            "values": {"default": {"mode": "constant", "value": 5.0}},
        }
    )
    assert ds.shape == (5, 2, 1, 121)
    assert ds.field_shape == (11, 11)
    np.testing.assert_array_equal(ds[0], np.full((2, 1, 121), 5.0, dtype=np.float32))


def test_open_dataset_synthetic_random_is_reproducible() -> None:
    spec = dict(
        grid={"bbox": [4, 0, 0, 4], "resolution": 2.0},
        variables=3,
        dates={"start": "2020-01-01", "end": "2020-01-01", "frequency": "6h"},
        values={"default": {"mode": "random", "mean": 0.0, "std": 1.0}},
        seed=42,
    )
    first = open_dataset(synthetic=dict(spec))
    second = open_dataset(synthetic=dict(spec))
    np.testing.assert_array_equal(first[:], second[:])


def test_open_dataset_synthetic_index_roundtrip() -> None:
    ds = open_dataset(
        synthetic={
            "grid": {"bbox": [2, 0, 0, 2], "resolution": 2.0},  # 2x2 = 4 gridpoints
            "variables": 2,
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "1d"},
            "values": {"default": {"mode": "index"}},
        }
    )
    n_vars, n_ens, n_grid = 2, 1, 4
    for d in range(2):
        for v in range(2):
            for g in range(4):
                expected = ((d * n_vars + v) * n_ens + 0) * n_grid + g
                assert ds[d, v, 0, g] == expected


def test_open_dataset_synthetic_composes_with_select() -> None:
    # synthetic= is a drop-in for dataset=: open_dataset transform keywords
    # still apply to the result.
    ds = open_dataset(
        synthetic={
            "grid": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
            "variables": ["a", "b", "c"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "1d"},
        },
        select=["a", "c"],
    )
    assert ds.variables == ["a", "c"]
    assert ds.shape[1] == 2


def test_open_dataset_synthetic_composes_with_date_subsetting() -> None:
    spec = {
        "grid": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
        "variables": 2,
        "dates": {"start": "2020-01-01", "end": "2020-01-05", "frequency": "1d"},
    }
    assert len(open_dataset(synthetic=dict(spec))) == 5
    subset = open_dataset(synthetic=dict(spec), start="2020-01-02", end="2020-01-04")
    assert len(subset) == 3


def test_open_dataset_synthetic_composes_with_rename() -> None:
    ds = open_dataset(
        synthetic={
            "grid": {"bbox": [4, 0, 0, 4], "resolution": 2.0},
            "variables": ["a", "b"],
            "dates": {"start": "2020-01-01", "end": "2020-01-02", "frequency": "1d"},
        },
        rename={"a": "temperature"},
    )
    assert "temperature" in ds.variables
    assert "a" not in ds.variables


def test_open_dataset_synthetic_rejects_unknown_grid_key() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        open_dataset(
            synthetic={
                "grid": {"hexagon": 1},
                "variables": 1,
                "dates": {"start": "2020-01-01", "end": "2020-01-01", "frequency": "6h"},
            }
        )
