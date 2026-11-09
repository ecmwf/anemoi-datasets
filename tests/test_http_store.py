# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for directory listing on the HTTP store."""

import asyncio
from unittest.mock import patch

import pytest

from anemoi.datasets.compat import HTTPStore
from anemoi.datasets.compat import zarr_version

ROOT = "https://data.example/anemoi-datasets/era.zarr"
ROOT_DIR = ROOT + "/"
DATA_DIR = ROOT_DIR + "data/"

# Links fsspec reads from the server's directory pages, keyed by directory
PAGES = {
    ROOT_DIR: [
        ROOT_DIR,
        ROOT_DIR + "#httpsFormats",
        ROOT_DIR + ".zgroup",
        ROOT_DIR + "data/",
        ROOT_DIR + "dates/",
        "https://other.example/data",
    ],
    DATA_DIR: [
        DATA_DIR,
        DATA_DIR + ".zarray",
        DATA_DIR + "0.0.0.0",
        DATA_DIR + "0.0.0.0?C=N",
    ],
}


@pytest.mark.skipif(zarr_version >= 3, reason="requires zarr-python<3")
def test_listdir_returns_child_names() -> None:
    def ls(directory, detail=False):
        return PAGES[directory]

    with patch("fsspec.filesystem") as filesystem:
        filesystem.return_value.ls.side_effect = ls
        store = HTTPStore(ROOT)
        assert store.listdir() == [".zgroup", "data", "dates"]
        assert store.listdir("data") == [".zarray", "0.0.0.0"]


@pytest.mark.skipif(zarr_version < 3, reason="requires zarr-python>=3")
def test_list_dir_returns_child_names() -> None:
    async def ls(self, url, detail=True, **kwargs):
        # zarr-python 3 lists subdirectories without the trailing "/"
        return PAGES[url.rstrip("/") + "/"]

    async def list_dir(store, prefix):
        return [name async for name in store.list_dir(prefix)]

    with patch("fsspec.implementations.http.HTTPFileSystem._ls", ls):
        store = HTTPStore(ROOT)
        assert asyncio.run(list_dir(store, "")) == [".zgroup", "data", "dates"]
        assert asyncio.run(list_dir(store, "data")) == [".zarray", "0.0.0.0"]
