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

    group_keys = sorted(group.keys())
    if not group_keys:
        raise ValueError(f"Check group: {group} is empty.")

    for name in sorted(group_keys):
        if name.startswith("."):
            if verbosity > 1:
                LOG.info(f"Check group: skipping {name}")
            continue

        if isinstance(group[name], zarr.hierarchy.Group):
            _check_group(group[name], verbosity, *path, name)
        else:
            _check_array(group[name], verbosity, *path, name)


def _check_array(array, verbosity: int, *path) -> None:
    assert len(array.chunks) == len(array.shape)
    assert math.prod(array.shape) % math.prod(array.chunks) == 0

    file_count = math.prod(array.shape) // math.prod(array.chunks)

    full = os.path.join(*path)

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
