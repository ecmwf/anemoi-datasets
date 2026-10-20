# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for directory listing on the Zarr 2 HTTP store."""

from unittest.mock import patch

from anemoi.datasets.compat.zarr2 import HTTPStore

ROOT = "https://data.example/anemoi-datasets/era.zarr"


def test_listdir_returns_child_names() -> None:
    root = ROOT + "/"
    data = root + "data/"
    pages = {
        root: [
            root,
            root + "#httpsFormats",
            root + ".zgroup",
            root + "data/",
            root + "dates/",
            "https://other.example/data",
        ],
        data: [
            data,
            data + ".zarray",
            data + "0.0.0.0",
            data + "0.0.0.0?C=N",
        ],
    }

    def ls(directory, detail=False):
        return pages[directory]

    with patch("fsspec.filesystem") as filesystem:
        filesystem.return_value.ls.side_effect = ls
        store = HTTPStore(ROOT)
        assert store.listdir() == [".zgroup", "data", "dates"]
        assert store.listdir("data") == [".zarray", "0.0.0.0"]
