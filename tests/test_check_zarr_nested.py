# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``check_zarr`` on a Zarr format 2 store with the flat chunk layout (``0.0.0``) and the nested one (``0/0/0``)."""

import os

import numpy as np
import pytest
import zarr

from anemoi.datasets.compat import zarr_version
from anemoi.datasets.misc.check import check_zarr

pytestmark = pytest.mark.skipif(zarr_version >= 3, reason="NestedDirectoryStore only exists in zarr 2")


def _write(path: str, nested: bool) -> None:
    store = zarr.storage.NestedDirectoryStore(path) if nested else zarr.storage.DirectoryStore(path)
    root = zarr.group(store=store)
    root.create_dataset("data", data=np.arange(24, dtype="float32").reshape(4, 2, 3), chunks=(1, 2, 3))


@pytest.mark.parametrize("nested", [False, True])
def test_check_zarr_accepts_flat_and_nested_chunks(tmp_path, nested):
    path = str(tmp_path / "data.zarr")
    _write(path, nested)
    check_zarr(path)


def test_check_zarr_rejects_a_file_that_is_not_a_chunk(tmp_path):
    path = str(tmp_path / "data.zarr")
    _write(path, nested=False)
    open(os.path.join(path, "data", "not_a_chunk"), "w").close()
    with pytest.raises(ValueError, match="not a valid chunk file"):
        check_zarr(path)
