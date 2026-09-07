# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.datasets.create.recipe.output import GriddedOutput


def _coords(*, variables=101, ensembles=1, grid_points=100):
    return {
        "dates": range(10),
        "variables": range(variables),
        "ensembles": range(ensembles),
        "values": range(grid_points),
    }


def test_default_chunking_splits_grid_four_ways():
    assert GriddedOutput().get_chunking(_coords(grid_points=100)) == (1, 101, 1, 100)


def test_default_chunking_handles_non_divisible_grid():
    assert GriddedOutput().get_chunking(_coords(grid_points=101)) == (1, 101, 1, 101)


@pytest.mark.parametrize(
    ("name", "grid_points", "expected_chunks"),
    [
        ("o96", 40_320, 1),
        ("n320", 542_080, 4),
        ("o1280", 6_599_680, 4),
        ("o2560", 26_306_560, 8),
    ],
)
def test_default_grid_chunking(name, grid_points, expected_chunks):
    chunks = GriddedOutput().get_chunking(_coords(grid_points=grid_points))

    print(f"{name}: {grid_points:,} grid points -> " f"{expected_chunks} chunks of {chunks[-1]:,} grid points")

    assert grid_points // chunks[-1] == expected_chunks


def test_default_chunking_increases_chunking_for_large_dates():
    output = GriddedOutput()
    # Four grid cells per chunk would exceed the signed 32-bit codec limit;
    # eight grid splits produce chunks containing two cells instead.
    assert output.get_chunking(_coords(variables=200_000_000, grid_points=16)) == (1, 200_000_000, 1, 2)


def test_default_chunking_decreases_chunking_for_small_dates():
    # Four grid splits would create 16 MiB chunks. Two grid splits produce
    # 32 MiB chunks, matching the configured minimum.
    assert GriddedOutput().get_chunking(_coords(grid_points=166_112)) == (1, 101, 1, 83_056)


def test_default_chunking_explicit_grid_chunking_works():
    output = GriddedOutput(chunking={"dates": 1, "ensembles": 1, "values": 7})
    assert output.get_chunking(_coords(grid_points=100)) == (1, 101, 1, 7)


def test_default_chunking_errors_when_one_grid_point_exceeds_codec_limit():
    output = GriddedOutput()
    with pytest.raises(ValueError, match="single-grid-point chunk"):
        output.get_chunking(_coords(variables=600_000_000, grid_points=4))
