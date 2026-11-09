# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import os

import numpy as np
import pytest
import yaml

from anemoi.datasets import open_dataset
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.compat import zarr_version
from anemoi.datasets.create.tasks import run_task

ZARR_FORMATS = [
    2,
    pytest.param(3, marks=pytest.mark.skipif(zarr_version < 3, reason="Zarr format 3 needs zarr-python>=3")),
]

NDATES = 4
NPOINTS = 10
VARIABLES = ["a", "b"]


def _write_source(path: str) -> None:
    """Write a tiny anemoi dataset (the source of the recipe below) to ``path``."""
    dates = np.array([np.datetime64("2021-01-01T00:00") + np.timedelta64(6 * i, "h") for i in range(NDATES)], "<M8[s]")
    root = zarr_open(path, "w")
    root.create_array("data", data=np.random.default_rng(0).random((NDATES, len(VARIABLES), 1, NPOINTS), "float32"))
    root.create_array("dates", data=dates)
    root.create_array("latitudes", data=np.linspace(-10, 10, NPOINTS))
    root.create_array("longitudes", data=np.linspace(0, 20, NPOINTS))
    root.attrs["frequency"] = "6h"
    root.attrs["resolution"] = "o96"
    root.attrs["name_to_index"] = {v: i for i, v in enumerate(VARIABLES)}
    root.attrs["data_request"] = {"grid": 1, "area": "g", "param_level": {}}
    root.attrs["variables_metadata"] = {v: {} for v in VARIABLES}


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_created_data_array_uses_blosc_lz4(tmp_path, zarr_format) -> None:
    """The ``data`` array is written with Blosc(lz4, clevel 5, byte shuffle) in every zarr format."""
    source = str(tmp_path / "source.zarr")
    _write_source(source)

    recipe = {
        "dates": {"start": "2021-01-01T00:00:00", "end": "2021-01-01T18:00:00", "frequency": "6h"},
        "input": {"anemoi_dataset": {"dataset": source}},
        "output": {"format": f"zarr{zarr_format}"},
    }
    recipe_path = str(tmp_path / "recipe.yaml")
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    # The tasks are run directly: tests/create/utils/create.py asserts the output is in Zarr format 2
    output = str(tmp_path / "out.zarr")
    run_task("init", recipe=recipe_path, path=output, overwrite=True)
    run_task("load", path=output)
    run_task("finalise", path=output)

    if zarr_format == 2:
        with open(os.path.join(output, "data", ".zarray")) as f:
            compressor = json.load(f)["compressor"]
        assert compressor["id"] == "blosc"
        assert (compressor["cname"], compressor["clevel"], compressor["shuffle"]) == ("lz4", 5, 1)
    else:
        with open(os.path.join(output, "data", "zarr.json")) as f:
            codecs = {c["name"]: c["configuration"] for c in json.load(f)["codecs"]}
        assert "zstd" not in codecs
        assert (codecs["blosc"]["cname"], codecs["blosc"]["clevel"], codecs["blosc"]["shuffle"]) == (
            "lz4",
            5,
            "shuffle",
        )

    np.testing.assert_array_equal(open_dataset(output)[:], open_dataset(source)[:])
