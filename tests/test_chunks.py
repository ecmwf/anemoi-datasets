# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Test suite for the ChunkFilter class in the anemoi.datasets.create.chunks module."""

import pytest

from anemoi.datasets.create.chunks import ChunkFilter


def test_chunk_filter():
    """Test the ChunkFilter class with various inputs and scenarios.

    Test cases:
    1. No filter applied.
    2. Invalid input for parts.
    3. Parts as a string representation of a fraction.
    4. Iteration over the filtered chunks.
    """
    # Test case 1: no filter
    cf = ChunkFilter(parts=None, total=10)
    assert cf(5) is True
    assert len(list(cf)) == len(cf)

    cf = ChunkFilter(parts="", total=10)
    assert cf(5) is True
    assert len(list(cf)) == len(cf)

    cf = ChunkFilter(parts=[], total=10)
    assert cf(5) is True
    assert len(list(cf)) == len(cf)

    # Test case 2: wrong input
    with pytest.raises(AssertionError):
        cf = ChunkFilter(parts="4/3", total=10)

    cf = ChunkFilter(parts="1/3", total=10)
    with pytest.raises(AssertionError):
        cf(-1)
    with pytest.raises(AssertionError):
        cf(10)

    # Test case 3: parts is a string representation of fraction
    cf = ChunkFilter(parts="1/3", total=10)
    cf_ = ChunkFilter(parts=["1/3"], total=10)
    assert cf(0) is cf_(0) is True
    assert cf(1) is cf_(1) is True
    assert cf(2) is cf_(2) is True
    assert cf(3) is cf_(3) is True
    assert cf(4) is cf_(4) is False
    assert cf(5) is cf_(5) is False
    assert cf(6) is cf_(6) is False
    assert cf(7) is cf_(7) is False
    assert cf(8) is cf_(8) is False
    assert cf(9) is cf_(9) is False
    assert len(list(cf)) == len(cf)

    cf = ChunkFilter(parts="2/3", total=10)
    cf_ = ChunkFilter(parts=["2/3"], total=10)
    assert cf(0) is cf_(0) is False
    assert cf(1) is cf_(1) is False
    assert cf(2) is cf_(2) is False
    assert cf(3) is cf_(3) is False
    assert cf(4) is cf_(4) is True
    assert cf(5) is cf_(5) is True
    assert cf(6) is cf_(6) is True
    assert cf(7) is cf_(7) is False
    assert cf(8) is cf_(8) is False
    assert cf(9) is cf_(9) is False
    assert len(list(cf)) == len(cf)

    cf = ChunkFilter(parts="3/3", total=10)
    cf_ = ChunkFilter(parts=["3/3"], total=10)
    assert cf(0) is cf_(0) is False
    assert cf(1) is cf_(1) is False
    assert cf(2) is cf_(2) is False
    assert cf(3) is cf_(3) is False
    assert cf(4) is cf_(4) is False
    assert cf(5) is cf_(5) is False
    assert cf(6) is cf_(6) is False
    assert cf(7) is cf_(7) is True
    assert cf(8) is cf_(8) is True
    assert cf(9) is cf_(9) is True
    assert len(list(cf)) == len(cf)

    # Test case 4: test __iter__
    cf = ChunkFilter(parts="2/3", total=10)
    for i in cf:
        assert cf(i) is True
    assert len(list(cf)) == 3


if __name__ == "__main__":
    """Run all test functions in the module."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
