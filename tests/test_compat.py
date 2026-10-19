# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import anemoi.datasets
from anemoi.datasets.commands.copy import ZarrCopier
from anemoi.datasets.compat import DEFAULT_ZARR_FORMAT
from anemoi.datasets.compat import ZarrFileNotFoundError
from anemoi.datasets.compat import blosc_compressor
from anemoi.datasets.compat import get_zarr_format
from anemoi.datasets.compat import zarr_format_from_name
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.compat import zarr_version
from anemoi.datasets.create.dataset import Dataset
from anemoi.datasets.misc.check import check_zarr

zarr2_only = pytest.mark.skipif(zarr_version >= 3, reason="requires zarr-python<3")
zarr3_only = pytest.mark.skipif(zarr_version < 3, reason="requires zarr-python>=3")


def _format_on_disk(path: Path) -> int:
    """Return the Zarr format of the group or array at ``path`` from its metadata files."""
    v2 = (path / ".zgroup").exists() or (path / ".zarray").exists()
    v3 = (path / "zarr.json").exists()
    assert v2 != v3, f"{path} has {'both' if v2 else 'no'} Zarr format 2 and 3 metadata"
    return 2 if v2 else 3


def test_default_zarr_format() -> None:
    assert DEFAULT_ZARR_FORMAT == 2


@pytest.mark.parametrize("mode", ["w", "w-", "a"])
def test_zarr_open_creates_default_format(tmp_path: Path, mode: str) -> None:
    path = tmp_path / "test.zarr"

    root = zarr_open(str(path), mode)
    root.create_group("group").create_array("array", shape=(4,), dtype="int64")

    assert get_zarr_format(root) == DEFAULT_ZARR_FORMAT
    assert _format_on_disk(path) == DEFAULT_ZARR_FORMAT
    assert _format_on_disk(path / "group") == DEFAULT_ZARR_FORMAT
    assert _format_on_disk(path / "group" / "array") == DEFAULT_ZARR_FORMAT


def test_zarr_open_append_keeps_existing_store(tmp_path: Path) -> None:
    path = str(tmp_path / "test.zarr")
    zarr_open(path, "w").attrs["key"] = "value"

    assert zarr_open(path, "a").attrs["key"] == "value"


@pytest.mark.parametrize("mode", ["r", "r+"])
def test_zarr_open_missing_store(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ZarrFileNotFoundError):
        zarr_open(str(tmp_path / "missing.zarr"), mode)


def test_zarr_open_unsupported_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        zarr_open(str(tmp_path / "test.zarr"), "w+")


def test_zarr_open_checks_format_of_existing_store(tmp_path: Path) -> None:
    path = str(tmp_path / "test.zarr")
    zarr_open(path, "w")

    assert get_zarr_format(zarr_open(path, "r", zarr_format=2)) == 2
    with pytest.raises(ValueError, match="expected format 3"):
        zarr_open(path, "r", zarr_format=3)


@zarr2_only
def test_zarr_open_format_3_requires_zarr3(tmp_path: Path) -> None:
    path = tmp_path / "test.zarr"

    with pytest.raises(ValueError, match="requires zarr-python>=3"):
        zarr_open(str(path), "w", zarr_format=3)

    assert not path.exists()


@zarr3_only
def test_zarr_open_creates_format_3(tmp_path: Path) -> None:
    path = tmp_path / "test.zarr"

    zarr_open(str(path), "w", zarr_format=3).create_array("array", shape=(4,), dtype="int64")

    assert _format_on_disk(path) == 3
    assert _format_on_disk(path / "array") == 3


@zarr3_only
def test_zarr_open_append_does_not_mix_formats(tmp_path: Path) -> None:
    path = tmp_path / "test.zarr"
    zarr_open(str(path), "w", zarr_format=3)

    with pytest.raises(ValueError, match="expected format 2"):
        zarr_open(str(path), "a", zarr_format=2)

    assert _format_on_disk(path) == 3


@pytest.mark.parametrize("zarr_format", [2, pytest.param(3, marks=zarr3_only)])
def test_blosc_compressor_matches_format(tmp_path: Path, zarr_format: int) -> None:
    path = tmp_path / "test.zarr"
    data = np.arange(10, dtype=np.float32)

    root = zarr_open(str(path), "w", zarr_format=zarr_format)
    array = root.create_array(
        "array",
        data=data,
        compressors=blosc_compressor(cname="zstd", clevel=3, shuffle=2, zarr_format=zarr_format),
    )

    np.testing.assert_array_equal(array[:], data)
    if zarr_format == 2:
        assert json.loads((path / "array" / ".zarray").read_text())["compressor"]["id"] == "blosc"
    else:
        codecs = json.loads((path / "array" / "zarr.json").read_text())["codecs"]
        assert "blosc" in [codec["name"] for codec in codecs]


@pytest.mark.parametrize("zarr_format", [2, pytest.param(3, marks=zarr3_only)])
def test_check_zarr_follows_store_format(tmp_path: Path, zarr_format: int) -> None:
    path = tmp_path / "test.zarr"

    root = zarr_open(str(path), "w", zarr_format=zarr_format)
    root.create_array("data", data=np.arange(10, dtype=np.float32).reshape(5, 2), chunks=(2, 2))

    check_zarr(str(path))


def test_dataset_writes_default_format(tmp_path: Path) -> None:
    path = tmp_path / "test.zarr"

    dataset = Dataset(str(path), create=True)
    dataset.add_array(
        name="data",
        shape=(2, 3),
        chunks=(1, 3),
        dtype="float32",
        fill_value=np.nan,
        dimensions=("time", "cell"),
    )
    dataset.add_array(name="_build/flags", data=np.zeros(2, dtype=bool), dimensions=("group",))

    for node in (path, path / "data", path / "_build", path / "_build" / "flags"):
        assert _format_on_disk(node) == DEFAULT_ZARR_FORMAT


@zarr3_only
def test_dataset_writes_format_3(tmp_path: Path) -> None:
    path = tmp_path / "test.zarr"

    dataset = Dataset(str(path), create=True, format="zarr3")
    dataset.add_array(name="_build/flags", data=np.zeros(2, dtype=bool), dimensions=("group",))

    for node in (path, path / "_build", path / "_build" / "flags"):
        assert _format_on_disk(node) == 3

    # Updating detects the format of the existing store
    assert get_zarr_format(Dataset(str(path), update=True).store) == 3


def test_zarr_format_from_name() -> None:
    assert zarr_format_from_name("zarr2") == 2
    assert zarr_format_from_name("zarr3") == 3
    with pytest.raises(ValueError, match="Invalid format 'zarr4'"):
        zarr_format_from_name("zarr4")


def test_dataset_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "test.zarr"

    with pytest.raises(ValueError, match="Invalid format"):
        Dataset(str(path), create=True, format="zarr4")

    assert not path.exists()


def test_dataset_update_requires_existing_store(tmp_path: Path) -> None:
    with pytest.raises(ZarrFileNotFoundError):
        Dataset(str(tmp_path / "missing.zarr"), update=True)


def test_copy_resharding_requires_zarr_format_3() -> None:
    with pytest.raises(ValueError, match="Resharding requires Zarr format 3"):
        ZarrCopier(
            source="source.zarr",
            target="target.zarr",
            transfers=1,
            block_size=None,
            overwrite=False,
            resume=False,
            verbosity=0,
            nested=False,
            obfuscate=False,
            rechunk=None,
            reshard="10",
            format="zarr2",
        )


# Module-level zarr functions that open or create stores, bypassing ``zarr_open``
_ZARR_OPENERS = {
    "open",
    "open_group",
    "open_array",
    "open_consolidated",
    "group",
    "create",
    "create_array",
    "create_group",
}


def test_stores_are_only_opened_with_zarr_open() -> None:
    """New stores must be written in a controlled Zarr format, which only ``zarr_open`` guarantees."""
    package = Path(anemoi.datasets.__file__).parent

    offenders = []
    for file in sorted(package.rglob("*.py")):
        if "compat" in file.relative_to(package).parts:
            continue
        for node in ast.walk(ast.parse(file.read_text(), filename=str(file))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "zarr"
                and node.func.attr in _ZARR_OPENERS
            ):
                offenders.append(f"{file.relative_to(package)}:{node.lineno}: zarr.{node.func.attr}()")

    assert not offenders, "Use anemoi.datasets.compat.zarr_open instead of:\n" + "\n".join(offenders)
