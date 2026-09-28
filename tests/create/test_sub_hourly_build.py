# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end sub-hourly builds, over a locally generated GRIB archive.

No network and no cached fixtures: the archive is written from an eccodes
sample, with steps encoded in minutes (GRIB edition 2, whose step keys carry
their own unit — ``"10m"``).  Each field's *values* say which field it is, so
the assertions prove that the right field landed in the right cell rather than
merely that the build ran.

Three builds, one per path that had to learn about sub-hourly steps:

* a gridded dataset on a 30-minute axis, mixing instants with 30-minute
  accumulations summed from a base-less 10-minute increment archive;
* a trajectory dataset with 10-minute steps;
* a trajectory dataset whose 30-minute accumulations are differences of
  from-zero accumulations of the imposed run.
"""

import datetime
import os

import numpy as np
import pytest

from anemoi.datasets import open_dataset

from .utils.create import create_dataset

eccodes = pytest.importorskip("eccodes")

BASE_TIME = datetime.datetime(2021, 1, 1, 0)

# A tiny global grid: 12 x 6 points is enough for statistics and chunking to
# be exercised without any of it being slow.
NI, NJ = 12, 6
NPOINTS = NI * NJ


def _sample_handle():
    """A regular lat/lon GRIB2 message, resized to the small test grid."""
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    eccodes.codes_set(h, "Ni", NI)
    eccodes.codes_set(h, "Nj", NJ)
    eccodes.codes_set(h, "latitudeOfFirstGridPointInDegrees", 60)
    eccodes.codes_set(h, "latitudeOfLastGridPointInDegrees", -60)
    eccodes.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0)
    eccodes.codes_set(h, "longitudeOfLastGridPointInDegrees", 330)
    eccodes.codes_set(h, "iDirectionIncrementInDegrees", 30)
    eccodes.codes_set(h, "jDirectionIncrementInDegrees", 24)
    return h


def _write_instant(path, valid_time, value, param_id=167):
    """An instantaneous field valid at *valid_time*, filled with *value*."""
    h = _sample_handle()
    try:
        eccodes.codes_set(h, "paramId", param_id)
        eccodes.codes_set(h, "dataDate", int(valid_time.strftime("%Y%m%d")))
        eccodes.codes_set(h, "dataTime", int(valid_time.strftime("%H%M")))
        eccodes.codes_set(h, "indicatorOfUnitOfTimeRange", 0)  # minutes
        eccodes.codes_set(h, "forecastTime", 0)
        eccodes.codes_set_values(h, np.full(NPOINTS, float(value)))
        with open(path, "wb") as f:
            eccodes.codes_write(h, f)
    finally:
        eccodes.codes_release(h)


def _write_forecast_instant(path, base, minutes, value, param_id=167):
    """An instantaneous field of the *base* run at a lead time of *minutes*."""
    h = _sample_handle()
    try:
        eccodes.codes_set(h, "paramId", param_id)
        eccodes.codes_set(h, "dataDate", int(base.strftime("%Y%m%d")))
        eccodes.codes_set(h, "dataTime", int(base.strftime("%H%M")))
        eccodes.codes_set(h, "indicatorOfUnitOfTimeRange", 0)  # minutes
        eccodes.codes_set(h, "forecastTime", minutes)
        eccodes.codes_set_values(h, np.full(NPOINTS, float(value)))
        with open(path, "wb") as f:
            eccodes.codes_write(h, f)
    finally:
        eccodes.codes_release(h)


def _write_accumulation(path, base, start_minutes, end_minutes, value, param_id=228228):
    """An accumulated field of the *base* run covering ``[start, end]`` minutes."""
    end = base + datetime.timedelta(minutes=end_minutes)
    h = _sample_handle()
    try:
        eccodes.codes_set(h, "productDefinitionTemplateNumber", 8)
        eccodes.codes_set(h, "paramId", param_id)
        eccodes.codes_set(h, "typeOfStatisticalProcessing", 1)  # accumulation
        eccodes.codes_set(h, "dataDate", int(base.strftime("%Y%m%d")))
        eccodes.codes_set(h, "dataTime", int(base.strftime("%H%M")))
        eccodes.codes_set(h, "indicatorOfUnitOfTimeRange", 0)  # minutes
        eccodes.codes_set(h, "forecastTime", start_minutes)
        eccodes.codes_set(h, "indicatorOfUnitForTimeRange", 0)  # minutes
        eccodes.codes_set(h, "lengthOfTimeRange", end_minutes - start_minutes)
        for key, part in (
            ("year", end.year),
            ("month", end.month),
            ("day", end.day),
            ("hour", end.hour),
            ("minute", end.minute),
            ("second", 0),
        ):
            eccodes.codes_set(h, f"{key}OfEndOfOverallTimeInterval", part)
        eccodes.codes_set_values(h, np.full(NPOINTS, float(value)))
        with open(path, "wb") as f:
            eccodes.codes_write(h, f)
    finally:
        eccodes.codes_release(h)


def _build(tmp_path, name, recipe):
    """Build *recipe* into a zarr under *tmp_path* and open it."""
    path = os.path.join(tmp_path, f"{name}.zarr")
    create_dataset(recipe=recipe, output=path)
    return open_dataset(path)


# ---------------------------------------------------------------------------
# gridded, 30-minute axis
# ---------------------------------------------------------------------------


def test_gridded_sub_hourly_dataset(tmp_path):
    """Instants every 30 min, plus 30-min accumulations from 10-min increments.

    ``2t`` carries the number of minutes since midnight, so its value proves
    the field's validity time; every 10-minute increment holds 10, so each
    30-minute window must accumulate to exactly 30.
    """
    tmp_path = str(tmp_path)
    inst_dir = os.path.join(tmp_path, "inst")
    inc_dir = os.path.join(tmp_path, "inc")
    os.makedirs(inst_dir)
    os.makedirs(inc_dir)

    times = [BASE_TIME + datetime.timedelta(minutes=10 * i) for i in range(19)]  # 00:00..03:00
    for t in times:
        _write_instant(
            os.path.join(inst_dir, f"inst{t:%Y%m%d%H%M}.grib"),
            t,
            t.hour * 60 + t.minute,
        )
        if t == times[0]:
            continue
        # A base-less 10-minute increment, addressed by the end of its window.
        window_start = t - datetime.timedelta(minutes=10)
        _write_accumulation(
            os.path.join(inc_dir, f"inc{t:%Y%m%d%H%M}.grib"),
            window_start,
            0,
            10,
            10,
        )

    recipe = {
        "dates": {"start": "2021-01-01 00:30:00", "end": "2021-01-01 03:00:00", "frequency": "30m"},
        "input": {
            "join": [
                {"grib": {"path": os.path.join(inst_dir, "inst{date:strftime(%Y%m%d%H%M)}.grib"), "param": ["2t"]}},
                {
                    "accumulate": {
                        "period": "30m",
                        "from": {"accumulation": "10m"},
                        "source": {
                            "grib": {
                                "path": os.path.join(inc_dir, "inc{end_date:strftime(%Y%m%d%H%M)}.grib"),
                                "param": ["tp"],
                            }
                        },
                    }
                },
            ]
        },
        "build": {"group_by": 1},
        "statistics": {"end": 2021},
    }

    ds = _build(tmp_path, "gridded", recipe)

    assert ds.frequency == datetime.timedelta(minutes=30)
    assert [str(d) for d in ds.dates] == [
        "2021-01-01T00:30:00",
        "2021-01-01T01:00:00",
        "2021-01-01T01:30:00",
        "2021-01-01T02:00:00",
        "2021-01-01T02:30:00",
        "2021-01-01T03:00:00",
    ]

    data = ds[:]
    assert not np.isnan(data).any()

    instant = next(v for v in ds.variables if v.startswith("2t"))
    np.testing.assert_array_equal(
        data[:, ds.variables.index(instant), 0, 0],
        np.array([30, 60, 90, 120, 150, 180], dtype=float),
    )
    np.testing.assert_allclose(data[:, ds.variables.index("tp"), 0, 0], 30.0)


# ---------------------------------------------------------------------------
# trajectories, 10-minute steps
# ---------------------------------------------------------------------------


def test_trajectory_sub_hourly_steps(tmp_path):
    """A 10-minute step axis, with each field's value naming its own cell."""
    tmp_path = str(tmp_path)
    archive = os.path.join(tmp_path, "fc")
    os.makedirs(archive)

    bases = [BASE_TIME, BASE_TIME + datetime.timedelta(hours=6)]
    minutes = [0, 10, 20, 30, 40, 50, 60]
    for base in bases:
        for m in minutes:
            _write_forecast_instant(
                os.path.join(archive, f"fc{base:%Y%m%d%H}+{m:04d}.grib"),
                base,
                m,
                base.hour * 1000 + m,
            )

    recipe = {
        "base_dates": {"start": "2021-01-01 00:00:00", "end": "2021-01-01 06:00:00", "frequency": "6h"},
        "steps": {"start": "0h", "end": "1h", "frequency": "10m"},
        "input": {
            "grib": {
                "path": os.path.join(archive, "fc{base_date:strftime(%Y%m%d%H)}+{step_minutes:int(%04d)}.grib"),
                "param": ["2t"],
            }
        },
        "output": {"layout": "trajectories"},
        "build": {"group_by": 1},
        "statistics": {"end": 2021},
    }

    ds = _build(tmp_path, "trajectory", recipe)

    assert ds.shape == (2, 1, 1, 7, NPOINTS)
    assert ds.step_frequency == datetime.timedelta(minutes=10)
    np.testing.assert_array_equal(
        ds.steps.astype("timedelta64[m]").astype(int),
        np.array(minutes),
    )

    data = ds[:]
    assert not np.isnan(data).any()
    # value = base hour * 1000 + step minutes, so this is the placement proof.
    np.testing.assert_array_equal(
        data[:, 0, 0, :, 0],
        np.array([[b.hour * 1000 + m for m in minutes] for b in bases], dtype=float),
    )

    # The reading side addresses a sub-hourly step as a frequency.
    path = os.path.join(tmp_path, "trajectory.zarr")
    assert open_dataset(path, step="30m").shape == (2, 1, 1, NPOINTS)
    narrowed = open_dataset(path, step_start="10m", step_end="40m", step_frequency="10m")
    assert narrowed.shape == (2, 1, 1, 4, NPOINTS)


# ---------------------------------------------------------------------------
# trajectories, sub-hourly accumulations
# ---------------------------------------------------------------------------


def test_trajectory_sub_hourly_accumulation(tmp_path):
    """30-minute windows differenced out of a from-zero 10-minute archive.

    The archive holds ``a(0, m) = m``, so *every* window ``[a, b]`` must come
    out as ``b − a`` — 30 for a 30-minute period, whatever the lead time.
    """
    tmp_path = str(tmp_path)
    archive = os.path.join(tmp_path, "tp")
    os.makedirs(archive)

    bases = [BASE_TIME, BASE_TIME + datetime.timedelta(hours=6)]
    for base in bases:
        for m in range(10, 121, 10):
            _write_accumulation(os.path.join(archive, f"tp{base:%Y%m%d%H}+{m:04d}.grib"), base, 0, m, m)

    recipe = {
        "base_dates": {"start": "2021-01-01 00:00:00", "end": "2021-01-01 06:00:00", "frequency": "6h"},
        "steps": {"start": "30m", "end": "2h", "frequency": "30m"},
        "input": {
            "accumulate": {
                "period": "30m",
                "from": {"base_dates": "from-layout", "steps": "from-layout", "accumulation": "from-zero"},
                "source": {
                    "grib": {
                        "path": os.path.join(archive, "tp{base_date:strftime(%Y%m%d%H)}+{step_minutes:int(%04d)}.grib"),
                        "param": ["tp"],
                    }
                },
            }
        },
        "output": {"layout": "trajectories"},
        "build": {"group_by": 1},
        "statistics": {"end": 2021, "allow_nans": True},
    }

    ds = _build(tmp_path, "trajectory-accum", recipe)

    assert ds.shape == (2, 1, 1, 4, NPOINTS)
    np.testing.assert_array_equal(
        ds.steps.astype("timedelta64[m]").astype(int),
        np.array([30, 60, 90, 120]),
    )

    data = ds[:]
    assert not np.isnan(data).any()
    np.testing.assert_allclose(data[:, 0, 0, :, 0], 30.0)
