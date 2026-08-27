# (C) Copyright 2024-2026 Anemoi contributors.
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
import zarr

from anemoi.datasets.commands import compare as compare_module
from anemoi.datasets.commands.compare import compare_anemoi_datasets


def _make_store(path: str, data: np.ndarray, chunks: tuple) -> None:
    """Create a minimal zarr store with a single "data" array."""
    group = zarr.open_group(path, mode="w")
    group.create_dataset("data", data=data, chunks=chunks, dtype=data.dtype)


@pytest.fixture
def datasets(tmp_path):
    data = np.arange(24 * 4 * 5, dtype=np.float64).reshape(24, 4, 5)

    reference = os.path.join(tmp_path, "reference.zarr")
    actual = os.path.join(tmp_path, "actual.zarr")

    # Chunked with a leading dimension of size 1, as is common for anemoi datasets
    _make_store(reference, data, chunks=(1, 4, 5))
    _make_store(actual, data.copy(), chunks=(1, 4, 5))

    return reference, actual, data


def test_compare_identical_datasets(datasets) -> None:
    reference, actual, _ = datasets

    errors = compare_anemoi_datasets(reference, actual, data=True)

    assert not errors


def test_compare_detects_different_values(datasets) -> None:
    reference, actual, data = datasets

    group = zarr.open_group(actual, mode="a")
    group["data"][10, 2, 3] = data[10, 2, 3] + 1000.0

    errors = compare_anemoi_datasets(reference, actual, data=True)

    assert errors
    assert any("different" in repr(e) for e in errors._errors)


def test_compare_with_small_buffer_subdivides_non_leading_dimensions(datasets, monkeypatch) -> None:
    """Force a tiny memory budget so the buffer shape must be subdivided along
    dimensions other than the (size 1) leading one, exercising the multi-axis
    task splitting logic."""
    reference, actual, _ = datasets

    monkeypatch.setattr(compare_module, "MAX_MEMORY_PER_WORKER", 256)

    errors = compare_anemoi_datasets(reference, actual, data=True)
    assert not errors

    group = zarr.open_group(actual, mode="a")
    group["data"][5, 1, 2] += 1.0

    errors = compare_anemoi_datasets(reference, actual, data=True)
    assert errors
