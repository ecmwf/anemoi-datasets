# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Callable
from functools import wraps
from unittest.mock import patch

import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.datasets import open_dataset


def _tests_zarrs(name: str) -> str:
    return f"https://anemoi-test.ecmwf.int/test-zarrs/{name}.zarr"


def zarr_tests(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("anemoi.datasets.data.stores.zarr_lookup", _tests_zarrs):
            return func(*args, **kwargs)

    return wrapper


def _test_dataset(ds):
    for p in ds.components():
        print(p)
        print(p.origins())


not_ready = pytest.mark.skip(reason="Not ready yet")


@skip_if_offline
@zarr_tests
@not_ready
def test_class_complement_none():
    pass
    # ds = open_dataset(
    #     source="cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
    #     complement="aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
    #     # adjust="all",
    # )


@skip_if_offline
@zarr_tests
@not_ready
def test_class_complement_nearest():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_concat():
    pass


@skip_if_offline
@zarr_tests
def test_class_number():
    ds = open_dataset(
        "aifs-ea-an-enda-0001-mars-o96-1979-2022-6h-v6",
        number=[1, 5, 6],
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_ensemble():
    ds = open_dataset(
        ensemble=[
            "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6",
            "aifs-ea-em-enda-0001-mars-o96-1979-2022-6h-v6",
        ]
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_missing_dates_fill():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_missing_dates_closest():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_missing_dates_interpolate():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_grids():
    ds = open_dataset(
        grids=[
            "cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
            "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        ],
        adjust="all",
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_cutout() -> None:
    ds = open_dataset(
        cutout=[
            "cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
            "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        ],
        adjust="all",
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_missing_date_error():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_interpolate_frequency():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_interpolate_nearest():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_join():
    pass


@skip_if_offline
@zarr_tests
def test_class_thinning():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
        thinning=100,
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_cropping():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
        area=[80, -10, 30, 40],
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_trim_edge():
    ds = open_dataset(
        "cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
        trim_edge=(1, 2, 3, 4),
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_merge():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_missing_dates():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_skip_missing_dates():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_missing_dataset():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_padded():
    pass


@skip_if_offline
@zarr_tests
def test_class_rescale_1():
    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        rescale={"2t": (1.0, -273.15)},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_rescale_2():
    try:
        import cfunits  # noqa: F401
    except FileNotFoundError:
        # cfunits requires the library udunits2 to be installed
        raise pytest.skip("udunits2 library not installed")

    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        rescale={"2t": ("K", "degC")},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_rescale_3():
    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        rescale={
            "2t": {"scale": 1.0, "offset": -273.15},
        },
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_select_select_1():
    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        select=["msl", "2t"],
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_select_select_2():
    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        select={"msl", "2t"},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_select_drop():
    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        drop=["2t", "msl"],
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_rename() -> None:
    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        rename={"2t": "temperature", "msl": "pressure"},
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_rename_with_overlap() -> None:
    ds = open_dataset(
        [
            {
                "dataset": "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
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
def test_class_statistics():
    pass


@skip_if_offline
@zarr_tests
def test_class_zarr():
    ds = open_dataset("aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6")
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_zarr_with_missing_dates():
    ds = open_dataset("rodeo-opera-files-o96-2013-2023-6h-v5")
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
def test_class_subset():
    ds = open_dataset(
        "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        frequency="12h",
        start=2017,
        end=2018,
    )
    _test_dataset(ds)


@skip_if_offline
@zarr_tests
@not_ready
def test_class_chain():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_zipbase():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_zip():
    pass


@skip_if_offline
@zarr_tests
@not_ready
def test_class_xy():
    pass


if __name__ == "__main__":
    test_class_rescale_2()
    exit(0)
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
