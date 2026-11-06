# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``anemoi-datasets copy --nested``."""

import os

import numpy as np
import pytest

from anemoi.datasets import open_dataset
from anemoi.datasets.commands.copy import ZarrCopier
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.compat import zarr_version

FORMATS = [2, 3] if zarr_version >= 3 else [2]

NDATES = 4
NPOINTS = 10
VARIABLES = ["a", "b"]
METADATA_FILES = {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}


def _write_source(path: str, zarr_format: int) -> np.ndarray:
    """Write a tiny anemoi dataset with one chunk per date, in the given Zarr format, and return its data."""
    dates = np.array([np.datetime64("2021-01-01T00:00") + np.timedelta64(6 * i, "h") for i in range(NDATES)], "<M8[s]")
    data = np.random.default_rng(0).random((NDATES, len(VARIABLES), 1, NPOINTS), "float32")
    root = zarr_open(path, "w", zarr_format=zarr_format)
    root.create_array("data", data=data, chunks=(1, len(VARIABLES), 1, NPOINTS))
    root.create_array("dates", data=dates)
    root.create_array("latitudes", data=np.linspace(-10, 10, NPOINTS))
    root.create_array("longitudes", data=np.linspace(0, 20, NPOINTS))
    root.attrs["frequency"] = "6h"
    root.attrs["resolution"] = "o96"
    root.attrs["name_to_index"] = {v: i for i, v in enumerate(VARIABLES)}
    root.attrs["data_request"] = {"grid": 1, "area": "g", "param_level": {}}
    root.attrs["variables_metadata"] = {v: {} for v in VARIABLES}
    root.attrs["start_date"] = "2021-01-01T00:00:00"
    root.attrs["end_date"] = "2021-01-01T18:00:00"
    return data


def _copy(source: str, target: str, nested: bool) -> None:
    ZarrCopier(
        source=source,
        target=target,
        transfers=1,
        block_size=1,
        overwrite=True,
        resume=False,
        verbosity=0,
        nested=nested,
        obfuscate=False,
        rechunk=None,
    ).run()


def _chunk_files(path: str) -> list[str]:
    """Paths (relative to the ``data`` array) of the chunk files, i.e. everything but metadata."""
    root = os.path.join(path, "data")
    found = []
    for dirpath, _dirs, files in os.walk(root):
        found += [os.path.relpath(os.path.join(dirpath, f), root) for f in files if f not in METADATA_FILES]
    return sorted(found)


def _is_nested(files: list[str]) -> bool:
    # flat chunk keys look like ``0.0.0.0``; nested ones are directories: ``0/0/0/0`` (format 3: ``c/0/0/0/0``)
    return bool(files) and all(os.sep in f and "." not in f for f in files)


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_copy_nested_layout_and_values(tmp_path, zarr_format):
    """``copy --nested`` writes chunks in subdirectories (the target keeps the source format) and opens through ``open_dataset``."""
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    expected = _write_source(source, zarr_format)

    _copy(source, target, nested=True)

    files = _chunk_files(target)
    assert _is_nested(files), files
    assert len(files) == NDATES

    np.testing.assert_array_equal(open_dataset(target)[:], expected)


def test_copy_without_nested_is_flat(tmp_path):
    """Without ``--nested`` a format 2 copy keeps the flat ``0.0.0.0`` chunk keys (format 3 is always nested)."""
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source(source, 2)
    _copy(source, target, nested=False)
    assert not _is_nested(_chunk_files(target))
