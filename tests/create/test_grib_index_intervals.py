# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np

from anemoi.datasets import open_dataset
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.grib_index import GribIndex
from anemoi.datasets.create.sources.grib_index import GribIndexSource

from .utils.create import create_dataset


class DummyContext:
    def trace(self, *args, **kwargs):
        pass


class CapturingGribIndexSource(GribIndexSource):
    def __init__(self, **kwargs):
        super().__init__(DummyContext(), indexdb="not-used.db", **kwargs)
        self.captured = None

    def _run_requests(self, full_requests):
        self.captured = full_requests
        return []


def test_grib_index_intervals_with_basetime_request() -> None:
    """Intervals with a basetime are addressed by their model run."""
    source = CapturingGribIndexSource(param=["tp"], levtype="sfc")

    base = datetime.datetime(2020, 1, 1, 0)
    interval = SignedInterval(
        start=base + datetime.timedelta(hours=23), end=base + datetime.timedelta(hours=24), base=base
    )
    source.execute_intervals(Intervals([interval.max], [interval]))

    assert source.captured == [
        ([interval.max], {"param": ["tp"], "levtype": "sfc", "date": 20200101, "time": 0}),
    ]

    base = datetime.datetime(2020, 1, 1, 12)
    interval = SignedInterval(start=base, end=base + datetime.timedelta(hours=1), base=base)
    source.execute_intervals(Intervals([interval.max], [interval]))

    assert source.captured[0][1]["time"] == 1200


def test_grib_index_intervals_without_basetime_request() -> None:
    """Intervals without a basetime are addressed by their length."""
    source = CapturingGribIndexSource(param=["tp"], levtype="sfc")

    start = datetime.datetime(2020, 1, 1, 0)
    interval = SignedInterval(start=start, end=start + datetime.timedelta(hours=3), base=None)
    source.execute_intervals(Intervals([interval.max], [interval]))

    assert source.captured == [([interval.max], {"param": ["tp"], "levtype": "sfc", "step": 3})]


def _write_forecast_accumulations(path: str, base: datetime.datetime, last_step: int) -> None:
    """Write hourly tp accumulations of one model run (step k holds k-1 to k, value k)."""
    import eccodes

    sample = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    eccodes.codes_set_key_vals(
        sample,
        {
            "Ni": 3,
            "Nj": 2,
            "latitudeOfFirstGridPointInDegrees": 1.0,
            "longitudeOfFirstGridPointInDegrees": 0.0,
            "latitudeOfLastGridPointInDegrees": 0.0,
            "longitudeOfLastGridPointInDegrees": 2.0,
            "iDirectionIncrementInDegrees": 1.0,
            "jDirectionIncrementInDegrees": 1.0,
        },
    )
    with open(path, "wb") as f:
        for step in range(1, last_step + 1):
            h = eccodes.codes_clone(sample)
            eccodes.codes_set_key_vals(
                h,
                {
                    "dataDate": int(base.strftime("%Y%m%d")),
                    "dataTime": int(base.strftime("%H%M")),
                    "paramId": 228,
                    "startStep": step - 1,
                    "endStep": step,
                },
            )
            eccodes.codes_set_values(h, np.full(6, float(step)))
            eccodes.codes_write(h, f)
            eccodes.codes_release(h)
    eccodes.codes_release(sample)


def test_accumulate_grib_index_with_basetime(tmp_path) -> None:
    """Accumulations from a forecast-indexed grib-index (step = lead time from a 00 UTC run)."""
    paths = []
    for day in (1, 2):
        path = str(tmp_path / f"tp-202001{day:02d}.grib")
        _write_forecast_accumulations(path, datetime.datetime(2020, 1, day), last_step=24)
        paths.append(path)

    indexdb = str(tmp_path / "grib-index.db")
    index = GribIndex(indexdb, update=True, overwrite=True)
    for path in paths:
        index.add_grib_file(path)

    recipe = {
        "dates": {"start": "2020-01-02T00:00:00", "end": "2020-01-02T03:00:00", "frequency": "3h"},
        "input": {
            "accumulate": {
                "period": "3h",
                "covering": {
                    "auto": {
                        "type": "accumulated-from-previous-step",
                        "basetime": [0],
                        "frequency": 1,
                        "last_step": 24,
                    }
                },
                "source": {"grib-index": {"indexdb": indexdb, "param": ["tp"], "levtype": "sfc"}},
            }
        },
    }

    ds = open_dataset(create_dataset(recipe=recipe, output=None))

    # 2020-01-02T00: steps 22-24 of the 2020-01-01 run; 2020-01-02T03: steps 1-3 of the 2020-01-02 run
    np.testing.assert_allclose(ds[0].squeeze(), 22 + 23 + 24)
    np.testing.assert_allclose(ds[1].squeeze(), 1 + 2 + 3)
