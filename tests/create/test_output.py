# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os

import numpy as np
import pytest
from pydantic import ValidationError

from anemoi.datasets import open_dataset
from anemoi.datasets.commands.recipe.migrate import migrate_output_format
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.compat import zarr_version
from anemoi.datasets.create.recipe import Recipe
from anemoi.datasets.create.recipe.output import GriddedOutput
from anemoi.datasets.create.recipe.output import TabularOutput

from .utils.create import create_dataset


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


def test_zarr_format_defaults_to_2():
    assert GriddedOutput().format == "zarr2"


@pytest.mark.skipif(zarr_version < 3, reason="requires zarr-python>=3")
def test_zarr_format_3_is_accepted_with_zarr3():
    assert GriddedOutput(format="zarr3").format == "zarr3"


@pytest.mark.skipif(zarr_version >= 3, reason="requires zarr-python<3")
def test_zarr_format_3_requires_zarr3():
    with pytest.raises(ValidationError, match="requires zarr-python>=3"):
        GriddedOutput(format="zarr3")


def test_zarr_format_must_be_2_or_3():
    with pytest.raises(ValidationError):
        GriddedOutput(format="zarr4")


def _recipe(output: dict) -> Recipe:
    return Recipe(input={"empty": {}}, dates={"values": []}, output=output)


def test_layout_selects_output():
    output = _recipe({"layout": "tabular"}).output
    assert isinstance(output, TabularOutput)
    assert output.format == "zarr2"


@pytest.mark.parametrize(("layout", "cls"), [("gridded", GriddedOutput), ("tabular", TabularOutput)])
def test_legacy_format_is_migrated_to_layout(layout, cls, caplog):
    # Recipes written before output.format selected the Zarr format, including those stored in existing datasets
    with caplog.at_level(logging.WARNING):
        output = _recipe({"format": layout}).output

    assert isinstance(output, cls)
    assert output.layout == layout
    assert output.format == "zarr2"
    assert f"'output.format: {layout}' is deprecated" in caplog.text


def test_legacy_format_conflicting_with_layout_is_rejected():
    with pytest.raises(ValidationError, match="conflicts with 'output.layout: gridded'"):
        _recipe({"format": "tabular", "layout": "gridded"})


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"format": "tabular"}, {"layout": "tabular"}),
        ({"format": "tabular", "dtype": "float64"}, {"layout": "tabular", "dtype": "float64"}),
        ({"format": "zarr3"}, {"format": "zarr3"}),
        ({"format": "tabular", "layout": "gridded"}, {"format": "tabular", "layout": "gridded"}),
    ],
)
def test_migrate_output_format(output, expected):
    assert migrate_output_format({"output": output}) == {"output": expected}


def test_invalid_compressor_is_rejected():
    for compressor in ({"id": "not-a-codec"}, {"id": "blosc", "not_an_option": 1}):
        with pytest.raises(ValidationError):
            GriddedOutput(compressor=compressor)


def _write_grib(path: str) -> None:
    """Write 2t on a 3x2 grid at 2020-01-01 00 and 06 UTC, with values equal to the hour."""
    import eccodes

    sample = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    eccodes.codes_set_key_vals(
        sample,
        {
            "Ni": 3,
            "Nj": 2,
            "latitudeOfFirstGridPointInDegrees": 1.0,
            "longitudeOfFirstGridPointInDegrees": 0.0,
            "latitudeOfLastGridPointInDegrees": 0.0,
            "longitudeOfLastGridPointInDegrees": 2.0,
            "iDirectionIncrementInDegrees": 1.0,
            "jDirectionIncrementInDegrees": 1.0,
        },
    )
    with open(path, "wb") as f:
        for hour in (0, 6):
            h = eccodes.codes_clone(sample)
            eccodes.codes_set_key_vals(h, {"dataDate": 20200101, "dataTime": hour * 100, "paramId": 167})
            eccodes.codes_set_values(h, np.full(6, float(hour)))
            eccodes.codes_write(h, f)
            eccodes.codes_release(h)
    eccodes.codes_release(sample)


def _compressor_on_disk(path: str) -> dict | None:
    with open(os.path.join(path, "data", ".zarray")) as f:
        return json.load(f)["compressor"]


def test_compressor_is_applied_to_data_array(tmp_path):
    grib = str(tmp_path / "2t.grib")
    _write_grib(grib)

    def create(name: str, output: dict) -> str:
        recipe = {
            "dates": {"start": "2020-01-01T00:00:00", "end": "2020-01-01T06:00:00", "frequency": "6h"},
            "input": {"grib": {"path": grib}},
            "output": output,
        }
        return create_dataset(recipe=recipe, output=str(tmp_path / f"{name}.zarr"))

    compressor = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}
    path = create("custom", {"compressor": compressor})
    assert _compressor_on_disk(path) == {**compressor, "blocksize": 0}
    np.testing.assert_allclose(open_dataset(path)[1].squeeze(), 6.0)

    # Without the option, the data array keeps the default compressor of zarr
    reference = str(tmp_path / "reference.zarr")
    zarr_open(reference, "w").create_array("data", shape=(1,), dtype="float32")
    assert _compressor_on_disk(create("default", {})) == _compressor_on_disk(reference)
