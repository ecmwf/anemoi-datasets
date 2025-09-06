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

from anemoi.utils.testing import skip_if_offline

from anemoi.datasets import open_dataset


def _tests_zarrs(name: str) -> str:
    return f"https://anemoi-test.ecmwf.int/test-zarrs/{name}.zarr"


def zarr_tests(func: Callable) -> Callable:
    """Decorator to mock the zarr_lookup function.

    Parameters
    ----------
    func : Callable
        Function to wrap.

    Returns
    -------
    Callable
        Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("anemoi.datasets.data.stores.zarr_lookup", _tests_zarrs):
            return func(*args, **kwargs)

    return wrapper


@skip_if_offline
@zarr_tests
def test_origins_rename() -> None:
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

    for p in ds.components():
        print(p)
        print(p.origins())


@skip_if_offline
@zarr_tests
def test_origins_cutout() -> None:
    ds = open_dataset(
        cutout=[
            "cerra-rr-an-oper-0001-mars-5p5km-1984-2020-6h-v2-hmsi",
            "aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6",
        ],
        adjust="all",
    )

    for p in ds.components():
        print(p)
        print(p.origins())


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
