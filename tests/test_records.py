# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import os

import numpy as np
import pytest

from anemoi.datasets.data import open_dataset
from anemoi.datasets.data.records import Record
from anemoi.datasets.data.records import Tabular


def check_numpy(x, y):
    assert x.shape == y.shape, f"Expected {x.shape} == {y.shape}"
    assert type(x) == type(y), f"Expected {type(x)} == {type(y)}"  # noqa: E721
    assert np.all(np.isnan(x) == np.isnan(y)) and np.all(
        np.nan_to_num(x) == np.nan_to_num(y)
    ), f"Expected {x} == {y} (ignoring NaNs)"


def _test(ds, nb_dates=None):
    grp = "metop_a_ascat"
    index_i = 0

    if nb_dates is not None:
        assert len(ds) == nb_dates, f"Expected {nb_dates} dates, got {len(ds)}"

    #################################
    # Order does not matter too much [i] and [grp] are exchangeable

    elt = ds[index_i]
    assert isinstance(elt, Record), (type(ds), type(elt))
    assert ds[index_i].dataset == ds, (type(ds[index_i].dataset), type(ds))

    group = ds[grp]
    assert isinstance(group, Tabular), type(group)

    x = ds[grp][index_i]
    y = ds[index_i][grp]
    check_numpy(x, y)

    ###############################################
    # lat and lon and timedelta are not the same for all elements
    # but they have the same size

    lat = ds[index_i].latitudes[grp]
    assert isinstance(lat, np.ndarray), type(lat)

    # Not implemented yet
    # lat = ds[grp].latitudes[index_i]
    # assert isinstance(lat, np.ndarray), type(lat)

    # Not implemented yet : do not need ?
    # lat = ds.latitudes[grp][index_i]
    # assert isinstance(lat, np.ndarray), type(lat)

    # Not implemented yet : do not need ?
    # lat = ds.latitudes[index_i][grp]
    # assert isinstance(lat, np.ndarray), type(lat)

    lon = ds[index_i].longitudes[grp]
    assert isinstance(lon, np.ndarray), type(lon)
    assert len(lat) == len(lon), f"Expected same size for lat and lon {len(lat)} == {len(lon)}"

    timedeltas = ds[index_i].timedeltas[grp]
    assert isinstance(timedeltas, np.ndarray), type(timedeltas)
    assert len(timedeltas) == len(lat), f"Expected same size for lat and timedeltas {len(lat)} == {len(timedeltas)}"

    #############################################
    # name_to_index is must be the same for all elements
    # name_to_index is a dict of dict (key is the group name)

    name_to_index = ds.name_to_index
    assert isinstance(name_to_index, dict), type(name_to_index)
    assert len(name_to_index) > 0, "name_to_index is empty"
    assert all(isinstance(k, str) for k in name_to_index.keys()), name_to_index
    assert all(isinstance(v, dict) for v in name_to_index.values()), name_to_index

    _name_to_index = ds[index_i].name_to_index
    assert list(name_to_index.keys()) == list(_name_to_index.keys()), (
        list(name_to_index.keys()),
        list(_name_to_index.keys()),
    )
    assert name_to_index == _name_to_index, "name_to_index is not the same for all elements"

    ###############################################
    # statistics is not the same for all elements
    # statistics is a dict of dict (first key is the group name)

    statistics = ds.statistics
    assert isinstance(statistics, dict), type(statistics)
    assert len(statistics) > index_i, "statistics is empty"
    assert all(isinstance(k, str) for k in statistics.keys()), statistics
    assert all(isinstance(v, dict) for v in statistics.values()), statistics
    assert grp in statistics, f"statistics does not contain {grp}"

    statistics_ = ds[grp].statistics
    assert isinstance(statistics_, dict), type(statistics_)
    assert "mean" in statistics_, "statistics does not contain mean"

    # ! here, the meaning could be ambigous, this is the statistics of the whole dataset.
    # Do not document this, and maybe remove it.
    _statistics = ds[index_i].statistics
    assert isinstance(_statistics, dict), type(_statistics)
    assert grp in _statistics, f"statistics does not contain {grp}"
    assert _statistics.keys() == ds.keys(), (_statistics.keys(), ds.keys())
    for group_name, stats in _statistics.items():
        assert "mean" in stats, f"statistics does not contain mean for {group_name}"
        for key, v in stats.items():
            assert np.all(statistics[group_name][key] == v), (key, statistics[group_name][key], v)

    assert statistics[grp].keys() == statistics_.keys(), (statistics[grp].keys(), statistics_.keys())
    for key, v in statistics[grp].items():
        assert np.all(statistics[grp][key] == v), (key, statistics[grp][key], v)


@pytest.mark.skipif(not os.path.exists("../../data/vz/obs-2018-11.vz"), reason="File not found")
def test_open():
    ds = open_dataset("../../data/vz/obs-2018-11.vz")
    _test(ds)


@pytest.mark.skipif(not os.path.exists("../../data/vz/obs-2018-11.vz"), reason="File not found")
def test_open_with_subset_dates():
    ds = open_dataset(
        "../../data/vz/obs-2018-11.vz",
        end="2018-11-30",
        select=[
            "metop_a_ascat.*",
            "amsr2_h180.rawbt_4",
            "amsr2_h180.rawbt_3",
        ],
    )
    _test(ds, nb_dates=8)


@pytest.mark.skipif(not os.path.exists("../../data/vz/obs-2018-11.vz"), reason="File not found")
def test_open_with_subset_select():
    ds = open_dataset(
        "../../data/vz/obs-2018-11.vz",
        select=[
            "amsr2_h180.rawbt_4",
            "amsr2_h180.rawbt_3",
            "metop_a_ascat.*",
        ],
    )
    _test(ds)


if __name__ == "__main__":

    test_open()
    test_open_with_subset_select()
    test_open_with_subset_dates()
