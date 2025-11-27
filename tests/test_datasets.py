# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.datasets.data import open_dataset


def _missing_credentials(url) -> bool:
    try:
        from anemoi.utils.remote.s3 import s3_options
    except ImportError:
        from anemoi.utils.remote.s3 import _s3_options as s3_options

    options = s3_options(url)
    return not options.get("access_key_id") or not options.get("secret_access_key")


@skip_if_offline
@pytest.mark.skipif(_missing_credentials("s3://ml-datasets/"), reason="No credentials for S3 access")
def test_s3_dataset() -> None:
    url = "s3://ml-datasets/aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6.zarr"
    ds = open_dataset(url)
    assert len(ds) == 64284


@skip_if_offline
def test_http_dataset() -> None:
    url = "https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr"
    ds = open_dataset(url)
    assert len(ds) == 65744


if __name__ == "__main__":
    test_s3_dataset()
