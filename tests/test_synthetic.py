# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest


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
