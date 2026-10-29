# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``mars`` metadata namespace of Xarray fields.

Consumers use this namespace to decide which fields describe *the same point
in space and time*.  In particular ``anemoi.transform``'s ``GroupByParam`` --
behind filters such as ``r-to-q``, ``r-to-d`` and ``uv-to-ddff`` -- groups a
field's components by it, and falls back to the *whole* metadata dictionary
when it is empty.  That fallback includes ``units``, which necessarily differs
between the components being combined (``K`` for a temperature, ``1`` for a
relative humidity), so an empty namespace puts every component in a group of
its own and the filter fails with "Missing component".
"""

import datetime

import numpy as np
import xarray as xr

from anemoi.datasets.create.sources.xarray_support import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support.metadata import XArrayMetadata

N_LAT, N_LON = 3, 4
TIME = datetime.datetime(2021, 1, 1, 12)

FLAVOUR = {
    "rules": {
        "latitude": {"name": "latitude"},
        "longitude": {"name": "longitude"},
        "time": {"name": "time"},
        "level": {"name": "level"},
    },
    "levtype": "pl",
}


def make_dataset() -> xr.Dataset:
    """Build a two-variable, two-level analysis dataset with differing units.

    ``t`` and ``r`` are deliberately given different ``units`` attributes,
    which is the situation that breaks grouping when the ``mars`` namespace is
    empty.

    Returns
    -------
    xr.Dataset
        A dataset with ``time``, ``level``, ``latitude`` and ``longitude``.
    """
    times = np.array([np.datetime64(TIME + datetime.timedelta(hours=3 * i), "ns") for i in range(2)])
    levels = np.array([850, 1000])
    shape = (len(times), len(levels), N_LAT, N_LON)

    ds = xr.Dataset(
        {
            "t": (("time", "level", "latitude", "longitude"), np.full(shape, 280.0)),
            "r": (("time", "level", "latitude", "longitude"), np.full(shape, 0.8)),
        },
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "level": ("level", levels, {"units": "hPa"}),
            "latitude": np.linspace(-60.0, 60.0, N_LAT),
            "longitude": np.linspace(0.0, 270.0, N_LON),
        },
    )
    ds["t"].attrs["units"] = "K"
    ds["r"].attrs["units"] = "1"
    return ds


def fieldlist() -> XarrayFieldList:
    """Return the test dataset as a field list."""
    return XarrayFieldList.from_xarray(make_dataset(), flavour=FLAVOUR)


def test_mars_namespace_is_populated() -> None:
    """``metadata(namespace="mars")`` describes the field."""
    field = fieldlist().sel(valid_datetime=TIME, param="t", level=850)[0]

    mars = field.metadata(namespace="mars")

    assert mars["param"] == "t"
    assert mars["levelist"] == 850
    assert mars["levtype"] == "pl"
    assert mars["date"] == 20210101
    assert mars["time"] == 1200
    assert mars["step"] == 0


def test_mars_namespace_excludes_units() -> None:
    """``units`` is not a MARS key, so it cannot split a grouping key."""
    field = fieldlist().sel(valid_datetime=TIME, param="t", level=850)[0]

    assert field.metadata("units") == "K"
    assert "units" not in field.metadata(namespace="mars")


def test_mars_namespace_holds_only_keys_the_field_has() -> None:
    """No MARS key is invented: the namespace is a subset of ``MARS_KEYS``."""
    field = fieldlist().sel(valid_datetime=TIME, param="t", level=850)[0]

    mars = field.metadata(namespace="mars")

    assert set(mars) <= set(XArrayMetadata.MARS_KEYS)
    # This dataset carries no ensemble dimension.
    assert "number" not in mars


def test_mars_namespace_agrees_across_components() -> None:
    """Two variables at the same point differ only by ``param``.

    This is precisely the grouping key the component-combining filters rely
    on, and it must not be perturbed by the variables' differing units.
    """
    keys = []
    for field in fieldlist().sel(valid_datetime=TIME, param=["t", "r"], level=850):
        mars = dict(field.metadata(namespace="mars"))
        mars.pop("param")
        keys.append(mars)

    assert len(keys) == 2
    assert keys[0] == keys[1]


def test_mars_namespace_distinguishes_levels() -> None:
    """Fields on different levels do not share a grouping key."""
    fs = fieldlist()

    at_850 = fs.sel(valid_datetime=TIME, param="t", level=850)[0].metadata(namespace="mars")
    at_1000 = fs.sel(valid_datetime=TIME, param="t", level=1000)[0].metadata(namespace="mars")

    assert at_850 != at_1000
    assert at_850["levelist"] == 850
    assert at_1000["levelist"] == 1000


def test_grouping_combines_components() -> None:
    """``GroupByParam`` finds both components of a conversion in one group.

    This is the end-to-end symptom: with an empty ``mars`` namespace this
    raises ``Missing component``.
    """
    from anemoi.transform.grouping import GroupByParam

    data = list(fieldlist().sel(valid_datetime=TIME, param=["r", "t"], level=850))

    groups = list(GroupByParam(["r", "t"]).iterate(data))

    assert len(groups) == 1
    assert sorted(f.metadata("param") for f in groups[0]) == ["r", "t"]
