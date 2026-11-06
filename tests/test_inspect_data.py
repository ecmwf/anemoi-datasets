# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import semantic_version

from anemoi.datasets.commands.inspect import Version
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.compat import zarr_version

ZARR_FORMATS = [
    2,
    pytest.param(3, marks=pytest.mark.skipif(zarr_version < 3, reason="Zarr format 3 needs zarr-python>=3")),
]


def _version(zarr_node) -> Version:
    # Bypass __init__, which needs a complete dataset; only the ``zarr`` attribute is used here.
    version = object.__new__(Version)
    version.zarr = zarr_node
    version.version = semantic_version.Version("0.21.0")
    return version


def _group_with_data(path, zarr_format):
    data = np.zeros((4, 2, 1, 8), dtype="float32")
    zarr_open(str(path), "w", zarr_format=zarr_format).create_array("data", data=data, chunks=(1, 2, 1, 8))
    return zarr_open(str(path), "r")


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_data_is_the_data_array_of_a_group(tmp_path, zarr_format):
    version = _version(_group_with_data(tmp_path / "ds.zarr", zarr_format))
    assert version.data.shape == (4, 2, 1, 8)
    assert version.shape == (4, 2, 1, 8)


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_data_falls_back_to_the_node_when_it_is_an_array(tmp_path, zarr_format):
    array = _group_with_data(tmp_path / "ds.zarr", zarr_format)["data"]
    version = _version(array)
    assert version.data is array
    assert version.shape == (4, 2, 1, 8)
