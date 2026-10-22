# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``check_zarr`` on a Zarr format 2 store with the flat chunk layout (``0.0.0``) and the nested one (``0/0/0``), and on a format 3 store."""

import os

import numpy as np
import pytest

from anemoi.datasets.compat import nested_chunk_kwargs
from anemoi.datasets.compat import nested_directory_store
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.compat import zarr_version
from anemoi.datasets.misc.check import check_zarr

LAYOUTS = [
    pytest.param(2, False, id="format2-flat"),
    pytest.param(2, True, id="format2-nested"),
    pytest.param(
        3,
        False,
        id="format3",
        marks=pytest.mark.skipif(zarr_version < 3, reason="Zarr format 3 needs zarr-python>=3"),
    ),
]


def _write(path: str, zarr_format: int, nested: bool) -> None:
    store = nested_directory_store(path) if nested else path
    kwargs = nested_chunk_kwargs(zarr_format) if nested else {}
    root = zarr_open(store, "w", zarr_format=zarr_format)
    root.create_array("data", data=np.arange(24, dtype="float32").reshape(4, 2, 3), chunks=(1, 2, 3), **kwargs)


@pytest.mark.parametrize(("zarr_format", "nested"), LAYOUTS)
def test_check_zarr_accepts_the_chunk_layouts(tmp_path, zarr_format, nested):
    path = str(tmp_path / "data.zarr")
    _write(path, zarr_format, nested)
    check_zarr(path)


@pytest.mark.parametrize("nested", [False, True])
def test_check_zarr_rejects_a_file_that_is_not_a_chunk(tmp_path, nested):
    path = str(tmp_path / "data.zarr")
    _write(path, 2, nested)
    open(os.path.join(path, "data", "not_a_chunk"), "w").close()
    with pytest.raises(ValueError, match="not a valid chunk file"):
        check_zarr(path)
