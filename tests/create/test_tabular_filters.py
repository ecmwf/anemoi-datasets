# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pandas as pd
import pytest

from anemoi.datasets.create.sources import tabular_add_gridded_fields as module


class _FakeField:
    def __init__(self, param, latitudes, longitudes, values):
        self.param = param
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.values = values

    def metadata(self, key):
        assert key == "param"
        return self.param

    def grid_points(self):
        return self.latitudes, self.longitudes

    def to_numpy(self, flatten=False):
        return self.values


@pytest.fixture
def fake_mars(monkeypatch):
    """Replace the MARS retrieval by a 3-point grid."""
    import earthkit.data as ekd

    latitudes = np.array([0.0, 45.0, -60.0])
    longitudes = np.array([0.0, 350.0, 120.0])
    fields = [
        _FakeField("lsm", latitudes, longitudes, np.array([0.0, 1.0, 0.5])),
        _FakeField("z", latitudes, longitudes, np.array([10.0, 20.0, 30.0])),
        # A second field for the same parameter is ignored
        _FakeField("lsm", latitudes, longitudes, np.array([9.0, 9.0, 9.0])),
    ]
    requests = []

    def from_source(name, request):
        assert name == "mars"
        requests.append(request)
        return fields

    monkeypatch.setattr(ekd, "from_source", from_source)
    module._load_fields.cache_clear()
    yield requests
    module._load_fields.cache_clear()


def test_add_gridded_fields(fake_mars) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-01 12:00"] * 4),
            "latitude": [1.0, 44.0, -59.0, 46.0],
            # -10 is the same longitude as 350, the nearest point must wrap around
            "longitude": [-1.0, -10.0, 121.0, 349.0],
        }
    )

    filter = module.AddGriddedFields(mars={"param": ["lsm", "z"], "grid": "n320"})
    result = filter.forward(frame)

    assert list(result.columns) == ["date", "latitude", "longitude", "lsm", "z"]
    np.testing.assert_array_equal(result["lsm"], [0.0, 1.0, 0.5, 1.0])
    np.testing.assert_array_equal(result["z"], [10.0, 20.0, 30.0, 20.0])
    assert "lsm" not in frame.columns

    # The fields are retrieved only once, even with a new filter instance
    module.AddGriddedFields(mars={"grid": "n320", "param": ["lsm", "z"]}, param=["z"]).forward(frame)
    assert len(fake_mars) == 1


def test_add_gridded_fields_empty_and_errors(fake_mars) -> None:
    empty = pd.DataFrame({"latitude": pd.Series(dtype=float), "longitude": pd.Series(dtype=float)})
    result = module.AddGriddedFields(mars={"param": "lsm"}).forward(empty)
    assert list(result.columns) == ["latitude", "longitude", "lsm", "z"]
    assert len(result) == 0

    with pytest.raises(ValueError, match="not returned"):
        module.AddGriddedFields(mars={"param": "lsm"}, param=["sdor"]).forward(empty)

    with pytest.raises(ValueError, match="already exist"):
        module.AddGriddedFields(mars={"param": "lsm"}).forward(empty.assign(lsm=1.0))


def test_add_gridded_fields_is_registered() -> None:
    from anemoi.datasets.create.input.action import _action_factories

    assert "add-gridded-fields" in _action_factories()


def test_daily_epochs_end_with_last_date() -> None:
    import datetime

    from anemoi.datasets.create.tabular.finalise import _daily_epochs

    first = datetime.datetime(2023, 12, 1, 12, 19, 55)
    last = datetime.datetime(2023, 12, 3, 11, 48, 27)
    epochs = _daily_epochs(first, last)
    assert list(epochs) == [
        np.datetime64("2023-12-01T12:19:55"),
        np.datetime64("2023-12-02T12:19:55"),
        np.datetime64("2023-12-03T11:48:27"),
    ]

    # No duplicate when the last date falls on the daily axis
    assert len(_daily_epochs(first, first + datetime.timedelta(days=2))) == 3
