# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.sources.accumulate.source import AccumulateSource


def _forecast_dates():
    bt = datetime.datetime(2025, 1, 1, 0)
    return ForecastDates([(bt + datetime.timedelta(hours=2), bt)])


def test_from_zero_rejected_for_netcdf_source():
    """xarray/NetCDF sources serve per-step increments, so from-zero must be rejected."""
    source = AccumulateSource(
        None,
        source={"netcdf": {"path": "x", "param": ["tp"]}},
        period="2h",
        accumulation="from-zero",
    )
    with pytest.raises(ValueError, match="from-zero.*not supported.*netcdf"):
        source.execute_forecast_dates(_forecast_dates())


def test_inner_is_xarray_classification():
    netcdf = AccumulateSource(
        None, source={"netcdf": {"path": "x", "param": ["tp"]}}, period="2h", accumulation="from-previous-step"
    )
    assert netcdf._inner_is_xarray() is True

    mars = AccumulateSource(None, source={"mars": {"param": ["tp"]}}, period="2h", accumulation="from-zero")
    assert mars._inner_is_xarray() is False
