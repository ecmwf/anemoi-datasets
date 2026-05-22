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
