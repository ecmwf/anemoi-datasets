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


def _coords(*, variables=4, ensembles=1, values=100):
    return {
        "dates": range(10),
        "variables": range(variables),
        "ensembles": range(ensembles),
        "values": range(values),
    }


def test_gridded_default_splits_grid_four_ways():
    assert GriddedOutput().get_chunking(_coords(values=100)) == (1, 4, 1, 25)


def test_gridded_default_handles_non_divisible_grid():
    assert GriddedOutput().get_chunking(_coords(values=101)) == (1, 4, 1, 26)


@pytest.mark.parametrize(
    ("name", "grid_points", "expected_points_per_chunk"),
    [
        ("o96", 40_320, 10_080),
        ("n320", 542_080, 135_520),
        ("o1280", 6_599_680, 1_649_920),
        ("o2560", 26_306_560, 6_576_640),
    ],
)
def test_default_grid_chunking(name, grid_points, expected_points_per_chunk):
    chunks = GriddedOutput().get_chunking(_coords(values=grid_points))

    print(
        f"{name}: {grid_points:,} grid points -> "
        f"4 chunks of {chunks[-1]:,} grid points"
    )

    assert chunks[-1] == expected_points_per_chunk
    assert grid_points // chunks[-1] == 4


def test_gridded_default_uses_power_of_two_grid_splits_for_large_chunks():
    output = GriddedOutput()
    # Four grid cells per chunk would exceed the signed 32-bit codec limit;
    # eight grid splits produce chunks containing two cells instead.
    assert output.get_chunking(_coords(variables=200_000_000, values=16)) == (1, 200_000_000, 1, 2)


def test_gridded_explicit_grid_chunking_is_preserved():
    output = GriddedOutput(chunking={"dates": 1, "ensembles": 1, "values": 7})
    assert output.get_chunking(_coords(values=100)) == (1, 4, 1, 7)


def test_gridded_raises_when_one_grid_point_exceeds_codec_limit():
    output = GriddedOutput()
    with pytest.raises(ValueError, match="single-grid-point chunk"):
        output.get_chunking(_coords(variables=600_000_000, values=4))