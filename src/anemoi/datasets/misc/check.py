# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# A collection of functions to support pytest testing

import logging
import math
import os
import re

LOG = logging.getLogger(__name__)


def _check_group(group, verbosity: int, *path) -> None:
    import zarr

    from anemoi.datasets.compat import zarr_private_files

    # Filter empty keys (https://github.com/zarr-developers/zarr-python/issues/3575)
    # and zarr-internal private entries.
    group_keys = sorted(k for k in group.keys() if k and k not in zarr_private_files)
    if not group_keys:
        raise ValueError(f"Check group: {group} is empty.")

    for name in group_keys:
        if name.startswith("."):
            if verbosity > 1:
                LOG.info(f"Check group: skipping {name}")
            continue

        if isinstance(group[name], zarr.Group):
            _check_group(group[name], verbosity, *path, name)
        else:
            _check_array(group[name], verbosity, *path, name)


def _check_array(array, verbosity: int, *path) -> None:
    from anemoi.datasets.compat import zarr_version

    assert len(array.chunks) == len(array.shape)

    full = os.path.join(*path)

    if zarr_version >= 3:
        # zarr v3: files live under c/ with slash-separated coordinates.
        # For sharded arrays the unit is the shard; otherwise the chunk.
        shards = getattr(array, "shards", None)
        outer = shards if shards is not None else array.chunks
        # ceil division: number of outer tiles per dimension
        file_count = math.prod(-(-s // o) for s, o in zip(array.shape, outer))

        chunk_dir = os.path.join(full, "c")
        if not os.path.isdir(chunk_dir):
            raise ValueError(f"Expected chunk directory {chunk_dir} not found for {array.name}.")

        count = sum(len(files) for _, _, files in os.walk(chunk_dir))

    else:
        # zarr v2: dot-separated chunk files directly in the array directory.
        assert math.prod(array.shape) % math.prod(array.chunks) == 0
        file_count = math.prod(array.shape) // math.prod(array.chunks)

        chunks = array.chunks
        count = 0
        for f in os.listdir(full):
            if verbosity > 1:
                LOG.info(f"Check array: checking {f}")

            if f.startswith("."):
                if verbosity > 1:
                    LOG.info(f"Check array: skipping {f}")
                continue

            bits = f.split(".")

            if len(bits) != len(chunks):
                raise ValueError(f"File {f} is not a valid chunk file.")

            if not all(re.match(r"^\d+$", bit) for bit in bits):
                raise ValueError(f"File {f} is not a valid chunk file.")

            count += 1
    if count != file_count:
        raise ValueError(f"File count {count} does not match expected {file_count} for {array.name}.")


def check_zarr(path: str, verbosity: int = 0) -> None:
    """Check if a Zarr archive is valid, that no files are missing, and that the chunking is correct.

    Parameters
    ----------
    path : str
        Path to the Zarr archive.
    verbosity : int, optional
        Verbosity level for logging. Default is 0 (no logging).
    """
    import zarr

    if verbosity > 0:
        LOG.info(f"Checking Zarr archive {path}")

    if not os.path.exists(path) and not os.path.isdir(path):
        # This does not work with non-directory Zarr archives
        raise ValueError(f"Path {path} does not exist.")

    _check_group(zarr.open(path, mode="r"), verbosity, path)
