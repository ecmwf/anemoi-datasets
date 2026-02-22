# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
from collections.abc import Callable
from functools import wraps
from unittest.mock import patch

import pytest
from anemoi.utils.testing import TEST_DATA_URL
from anemoi.utils.testing import skip_if_offline

from anemoi.datasets import open_dataset


def _tests_zarrs(name: str) -> str:
    return os.path.join(TEST_DATA_URL, "anemoi-datasets", f"{name}.zarr")


def zarr_tests(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("anemoi.datasets.usage.store.dataset_lookup", _tests_zarrs):
            return func(*args, **kwargs)

    return wrapper


def _test_dataset(ds, variables=None):

    if variables is not None:
        assert ds.variables == variables, (
            set(ds.variables) - set(variables),
            set(variables) - set(ds.variables),
            ds.variables,
        )

    # for p in ds.components():
    #     print(p)
    #     print(p.origins())


not_ready = pytest.mark.skip(reason="Not ready yet")


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_complement_none():
    pass
    # ds = open_dataset(
    #     source="cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
    #     complement="aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
    #     # adjust="all",
    # )


@skip_if_offline
@zarr_tests
def test_class_gridded_complement_nearest_1():

    ds = open_dataset(
        complement="cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        source="aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        interpolation="nearest",
    )
    _test_dataset(
        ds,
        variables=[
            "2t",
            "cos_latitude",
            "cp",
            "insolation",
            "lsm",
            "msl",
            "orog",
            "sf",
            "t_500",
            "t_850",
            "tp",
            "z",
            "z_500",
            "z_850",
        ],
    )


@skip_if_offline
@zarr_tests
def test_class_gridded_complement_nearest_2():
    ds = open_dataset(
        source="cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        complement="aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        interpolation="nearest",
    )
    _test_dataset(
        ds,
        variables=[
            "2t",
            "cos_latitude",
            "cp",
            "insolation",
            "lsm",
            "msl",
            "orog",
            "sf",
            "t_500",
            "t_850",
            "tp",
            "z",
            "z_500",
            "z_850",
        ],
    )


@skip_if_offline
@zarr_tests
def test_class_gridded_concat():
    ds = open_dataset(
        [
            "aifs-ea-an-oper-0001-mars-20p0-2016-2016-6h-v1",
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        ]
    )
    _test_dataset(
        ds,
        variables=[
            "2t",
            "cos_latitude",
            "cp",
            "insolation",
            "lsm",
            "msl",
            "t_500",
            "t_850",
            "tp",
            "z",
            "z_500",
            "z_850",
        ],
    )


@skip_if_offline
@zarr_tests
def test_class_gridded_number():
    ds = open_dataset(
        "aifs-ea-an-enda-0001-mars-20p0-2017-2017-6h-v1",
        members=[0, 2],
    )
    _test_dataset(
        ds,
        variables=[
            "2t",
            "cos_latitude",
            "cp",
            "insolation",
            "lsm",
            "msl",
            "t_500",
            "t_850",
            "tp",
            "z",
            "z_500",
            "z_850",
        ],
    )


@skip_if_offline
@zarr_tests
def test_class_gridded_ensemble():
    ds = open_dataset(
        ensemble=[
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
            "aifs-ea-em-enda-0001-mars-20p0-2017-2017-6h-v1",
        ]
    )
    _test_dataset(
        ds,
        variables=[
            "2t",
            "cos_latitude",
            "cp",
            "insolation",
            "lsm",
            "msl",
            "t_500",
            "t_850",
            "tp",
            "z",
            "z_500",
            "z_850",
        ],
    )


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_missing_dates_fill():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_missing_dates_closest():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_missing_dates_interpolate():
    pass


@skip_if_offline
@zarr_tests
def test_class_gridded_grids():
    ds = open_dataset(
        grids=[
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
            "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        ],
        adjust="all",
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_cutout() -> None:
    ds = open_dataset(
        cutout=[
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
            "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        ],
        adjust="all",
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_missing_date_error():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_interpolate_frequency():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_interpolate_nearest():
    pass


@skip_if_offline
@zarr_tests
def test_class_gridded_join_1():
    ds = open_dataset(
        [
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1-sfc",
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1-pl",
        ],
    )
    _test_dataset(ds, ["2t", "lsm", "msl", "z", "t_500", "t_850", "z_500", "z_850"])


@skip_if_offline
@zarr_tests
def test_class_gridded_join_2():
    ds = open_dataset(
        [
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1-pl",
            "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1-sfc",
        ],
    )
    _test_dataset(ds, ["t_500", "t_850", "z_500", "z_850", "2t", "lsm", "msl", "z"])


@skip_if_offline
@zarr_tests
def test_class_gridded_thinning_1():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        thinning=4,
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_thinning_2():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        method="distance-based",
        thinning=100,  # 100km
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_thinning_3():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        method="grid",
        thinning=100,  # 100km
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_thinning_4():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        method="random",
        thinning=0.5,  # 50% of points
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_cropping():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        area=[80, -10, 30, 40],
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_trim_edge():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p0-2017-2017-6h-v1",
        trim_edge=(1, 2, 3, 4),
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_merge():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_missing_dates():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_skip_missing_dates():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_missing_dataset():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_padded():
    pass


@skip_if_offline
@zarr_tests
def test_class_gridded_rescale_1():
    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        rescale={"2t": (1.0, -273.15)},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_rescale_2():
    try:
        import cfunits  # noqa: F401
    except FileNotFoundError:
        # cfunits requires the library udunits2 to be installed
        raise pytest.skip("udunits2 library not installed")

    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        rescale={"2t": ("K", "degC")},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_rescale_3():
    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        rescale={
            "2t": {"scale": 1.0, "offset": -273.15},
        },
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_select_select_1():
    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        select=["msl", "2t"],
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_select_select_2():
    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        select={"msl", "2t"},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_select_drop():
    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        drop=["2t", "msl"],
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_rename() -> None:
    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        rename={"2t": "temperature", "msl": "pressure"},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_rename_with_overlap() -> None:
    ds = open_dataset(
        [
            {
                "dataset": "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
                "select": ["cp", "tp"],
                "end": 2023,
                "frequency": "6h",
            },
            {
                "dataset": "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v1-precipitations",
                "end": 2023,
                "frequency": "6h",
                "rename": {"tp_0h_12h": "tp"},
                "select": ["tp_0h_12h"],
            },
        ],
        end=2022,
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_statistics():
    pass


@skip_if_offline
@zarr_tests
def test_class_gridded_zarr():
    ds = open_dataset("aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1")
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_zarr_with_missing_dates():
    ds = open_dataset("rodeo-opera-files-o96-2013-2023-6h-v5")
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_gridded_subset():
    ds = open_dataset(
        "aifs-ea-an-oper-0001-mars-20p0-2017-2017-6h-v1",
        frequency="12h",
        start=2017,
        end=2018,
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_chain():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_zipbase():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_zip():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_gridded_xy():
    pass


if __name__ == "__main__":
    from _pytest.outcomes import Skipped

    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            try:
                obj()
            except Skipped as e:
                print(f"Skipped {name}: {e}")
