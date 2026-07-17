# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest

from anemoi.datasets import open_dataset
from anemoi.datasets.usage.gridded.indexing import length_to_slices
from anemoi.datasets.usage.gridded.subset import Subset
from tests.test_data import mockup_open_zarr


def test_length_to_slices() -> None:
    """Test the length_to_slices function with various inputs."""
    lengths = [5, 7, 11, 13]
    datasets = [np.random.rand(n) for n in lengths]
    total = sum(lengths)

    combined = np.concatenate(datasets)

    for start in range(total):
        for stop in range(start, total):
            for step in range(1, stop - start + 1):
                index = slice(start, stop, step)
                print(index)
                slices = length_to_slices(index, lengths)
                result = [d[i] for (d, i) in zip(datasets, slices) if i is not None]
                result = np.concatenate(result)

                if (combined[index].shape != result.shape) or not (combined[index] == result).all():
                    print(index)
                    print(combined[index])
                    print(result)
                    print(slices)
                assert (combined[index] == result).all(), index


@mockup_open_zarr
def test_negative_indexing() -> None:
    """Test that negative integer indices work correctly."""
    ds = open_dataset("test-2021-2023-1h-o96-abcd", start=2022, end=2022)
    n = len(ds)

    # Negative integer index should return the same result as its positive equivalent
    for offset in [1, 2, 3]:
        neg_result = ds[-offset]
        pos_result = ds[n - offset]
        assert (neg_result == pos_result).all(), f"ds[-{offset}] != ds[{n - offset}]"

    # Negative index should produce non-empty result
    result = ds[-1]
    assert result.size > 0, "Negative index produced empty result"


@mockup_open_zarr
def test_subset_bounds_checking() -> None:
    """Test that out-of-range negative indices raise IndexError."""
    ds = open_dataset("test-2021-2023-1h-o96-abcd", start=2022, end=2022)
    assert isinstance(ds, Subset)

    n = len(ds)

    # These should work (boundary values)
    ds[-n]
    ds[n - 1]

    # These should raise IndexError
    with pytest.raises(IndexError):
        ds[-(n + 1)]
    with pytest.raises(IndexError):
        ds[n]


if __name__ == "__main__":
    test_length_to_slices()
