# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``copy`` gives the target data array the compressor of the source, in every combination of Zarr formats."""

import json
import os

import numcodecs
import numpy as np
import pytest

from anemoi.datasets.commands.copy import ZarrCopier
from anemoi.datasets.compat import blosc_compressor
from anemoi.datasets.compat import zarr_open
from anemoi.datasets.compat import zarr_version

NDATES = 4
NPOINTS = 10
VARIABLES = ["a", "b"]
FORMATS = [2, pytest.param(3, marks=pytest.mark.skipif(zarr_version < 3, reason="Zarr format 3 needs zarr-python>=3"))]
SHUFFLE = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}
BLOSC_LZ4 = ("blosc", "lz4", 5, "shuffle")


def _compressor_kwargs(spec: str, zarr_format: int) -> dict:
    """The ``create_array`` arguments that select the codec ``spec`` in the given format."""
    if spec == "default":
        return {}
    if spec == "none":
        return {"compressors": None}
    if spec == "blosc_lz4":
        return {"compressors": blosc_compressor(cname="lz4", clevel=5, shuffle=1, zarr_format=zarr_format)}
    if spec == "blosc_zstd_bitshuffle":
        return {"compressors": blosc_compressor(cname="zstd", clevel=3, shuffle=2, zarr_format=zarr_format)}
    if spec == "lz4":  # no format 3 equivalent
        return {"compressors": numcodecs.LZ4()}
    if zarr_format == 2:
        return {"compressors": {"zstd": numcodecs.Zstd(level=3), "gzip": numcodecs.GZip(level=4)}[spec]}
    from zarr import codecs

    return {"compressors": {"zstd": codecs.ZstdCodec(level=3), "gzip": codecs.GzipCodec(level=4)}[spec]}


def _write_source(path: str, zarr_format: int, spec: str, shards: bool = False) -> np.ndarray:
    """Write a tiny anemoi dataset whose data array uses the codec ``spec``, return its data."""
    dates = np.array([np.datetime64("2021-01-01T00:00") + np.timedelta64(6 * i, "h") for i in range(NDATES)], "<M8[s]")
    data = np.random.default_rng(0).random((NDATES, len(VARIABLES), 1, NPOINTS), "float32")
    root = zarr_open(path, "w", zarr_format=zarr_format)
    extra = {"shards": (2, len(VARIABLES), 1, NPOINTS)} if shards else {}
    root.create_array(
        "data",
        data=data,
        chunks=(1, len(VARIABLES), 1, NPOINTS),
        **extra,
        **_compressor_kwargs(spec, zarr_format),
    )
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


def _normalise(name: str, config: dict) -> tuple:
    if name == "blosc":
        shuffle = config["shuffle"]
        return ("blosc", config["cname"], config["clevel"], SHUFFLE[shuffle] if isinstance(shuffle, int) else shuffle)
    if name in ("zstd", "gzip"):
        return (name, config["level"])
    return (name,)


def _compressors(path: str) -> list:
    """The compressors of the ``data`` array, read from the metadata file: a list of tuples, empty for none."""
    array = os.path.join(path, "data")
    if os.path.exists(os.path.join(array, ".zarray")):
        with open(os.path.join(array, ".zarray")) as f:
            compressor = json.load(f)["compressor"]
        return [] if compressor is None else [_normalise(compressor["id"], compressor)]

    with open(os.path.join(array, "zarr.json")) as f:
        codecs = json.load(f)["codecs"]
    for codec in codecs:
        if codec["name"] == "sharding_indexed":
            codecs = codecs + codec["configuration"]["codecs"]
    return [
        _normalise(c["name"], c.get("configuration", {}))
        for c in codecs
        if c["name"] not in ("bytes", "sharding_indexed")
    ]


def _copy(source: str, target: str, zarr_format: int) -> None:
    ZarrCopier(
        source=source,
        target=target,
        transfers=1,
        block_size=1,
        overwrite=True,
        resume=False,
        verbosity=0,
        nested=False,
        obfuscate=False,
        rechunk=None,
        format=f"zarr{zarr_format}",
    ).run()


@pytest.mark.parametrize("spec", ["default", "none", "blosc_lz4", "blosc_zstd_bitshuffle", "zstd", "gzip"])
@pytest.mark.parametrize("target_format", FORMATS)
@pytest.mark.parametrize("source_format", FORMATS)
def test_copy_keeps_the_compressor_of_the_source(tmp_path, source_format, target_format, spec) -> None:
    source, target = str(tmp_path / "source.zarr"), str(tmp_path / "target.zarr")
    expected = _write_source(source, source_format, spec)
    wanted = _compressors(source)

    _copy(source, target, target_format)

    assert _compressors(target) == wanted
    np.testing.assert_array_equal(zarr_open(target, "r")["data"][:], expected)


@pytest.mark.skipif(zarr_version < 3, reason="the target has to be in format 3")
def test_copy_falls_back_to_blosc_lz4_when_the_codec_has_no_format_3_equivalent(tmp_path, caplog) -> None:
    source, target = str(tmp_path / "source.zarr"), str(tmp_path / "target.zarr")
    expected = _write_source(source, 2, "lz4")
    assert _compressors(source) == [("lz4",)]

    _copy(source, target, 3)

    assert _compressors(target) == [BLOSC_LZ4]
    assert "cannot translate" in caplog.text
    np.testing.assert_array_equal(zarr_open(target, "r")["data"][:], expected)


@pytest.mark.skipif(zarr_version < 3, reason="sharding needs zarr-python 3")
@pytest.mark.parametrize("target_format", [2, 3])
def test_copy_keeps_the_inner_compressor_of_a_sharded_source(tmp_path, target_format) -> None:
    source, target = str(tmp_path / "source.zarr"), str(tmp_path / "target.zarr")
    expected = _write_source(source, 3, "blosc_zstd_bitshuffle", shards=True)
    assert _compressors(source) == [("blosc", "zstd", 3, "bitshuffle")]

    _copy(source, target, target_format)

    assert _compressors(target) == [("blosc", "zstd", 3, "bitshuffle")]
    np.testing.assert_array_equal(zarr_open(target, "r")["data"][:], expected)
