# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Codecs written as in Zarr format 3 metadata (``{"name": ..., "configuration": {...}}``) and their Zarr format 2
(numcodecs) equivalents.
"""

from typing import Any

import numcodecs

BLOSC_SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")

# Defaults of the zarr-python 3 codecs, so that a codec gets the same settings in both formats
_DEFAULTS = {
    "blosc": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle", "blocksize": 0},
    "zstd": {"level": 0, "checksum": False},
    "gzip": {"level": 5},
}


def _value(x: Any) -> Any:
    return getattr(x, "value", x)


def numcodecs_codec(spec: dict[str, Any]) -> numcodecs.abc.Codec:
    """Return the numcodecs codec, for a Zarr format 2 array, equivalent to a Zarr format 3 codec.

    ``blosc``, ``zstd`` and ``gzip`` are translated. ``numcodecs.<id>`` is resolved with ``numcodecs.get_codec``,
    which includes codecs from other packages registered through the ``numcodecs.codecs`` entry point.

    Parameters
    ----------
    spec : dict
        The codec as in Zarr format 3 metadata, ``{"name": ..., "configuration": {...}}``.

    Returns
    -------
    numcodecs.abc.Codec
        The equivalent numcodecs codec.

    Raises
    ------
    NotImplementedError
        If the codec has no numcodecs equivalent.
    """
    name = spec["name"]
    config = {k: _value(v) for k, v in (spec.get("configuration") or {}).items()}

    if name.startswith("numcodecs."):
        config.pop("id", None)
        return numcodecs.get_codec({"id": name.removeprefix("numcodecs."), **config})

    if name not in _DEFAULTS:
        raise NotImplementedError(
            f"cannot translate codec {name!r} to Zarr format 2, use one of {', '.join(_DEFAULTS)} or 'numcodecs.<id>'"
        )
    config = {**_DEFAULTS[name], **config}

    if name == "blosc":
        config.pop("typesize", None)  # numcodecs takes it from the data
        if config["shuffle"] not in BLOSC_SHUFFLE:
            raise ValueError(f"Invalid blosc shuffle {config['shuffle']!r}, expected one of {BLOSC_SHUFFLE}")
        config["shuffle"] = BLOSC_SHUFFLE.index(config["shuffle"])
        return numcodecs.Blosc(**config)
    if name == "zstd":
        return numcodecs.Zstd(**config)
    return numcodecs.GZip(**config)


def zarr3_spec(codec: numcodecs.abc.Codec) -> dict[str, Any]:
    """Return the Zarr format 3 codec equivalent to a numcodecs Blosc, Zstd or GZip codec.

    Raises
    ------
    NotImplementedError
        If the codec has no built-in Zarr format 3 equivalent.
    """
    if isinstance(codec, numcodecs.Blosc) and codec.shuffle in (0, 1, 2):
        config = {"cname": codec.cname, "clevel": codec.clevel, "shuffle": BLOSC_SHUFFLE[codec.shuffle]}
        return {"name": "blosc", "configuration": {**config, "blocksize": codec.blocksize}}
    if isinstance(codec, numcodecs.Zstd):
        return {"name": "zstd", "configuration": {"level": codec.level, "checksum": getattr(codec, "checksum", False)}}
    if isinstance(codec, numcodecs.GZip):
        return {"name": "gzip", "configuration": {"level": codec.level}}
    raise NotImplementedError(f"cannot translate {codec!r} to Zarr format 3")


def zarr3_codec(spec: dict[str, Any]) -> Any:
    """Return the zarr-python 3 codec for a Zarr format 3 codec, looked up in the zarr codec registry.

    The registry includes the numcodecs wrappers (``numcodecs.<id>``) and codecs from other packages registered
    through the ``zarr.codecs`` entry point.
    """
    from zarr.registry import get_codec_class

    return get_codec_class(spec["name"]).from_dict({"configuration": {}, **spec})
