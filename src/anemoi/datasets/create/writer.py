# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import warnings

import numpy as np

LOG = logging.getLogger(__name__)


class CubesFilter:
    def __init__(self, *, parts, total):
        if parts is None:
            self.parts = None
            return

        if len(parts) == 1:
            part = parts[0]
            if part.lower() in ["all", "*"]:
                self.parts = None
                return

            if "/" in part:
                i_chunk, n_chunks = part.split("/")
                i_chunk, n_chunks = int(i_chunk), int(n_chunks)

                assert i_chunk > 0, f"Chunk number {i_chunk} must be positive."
                if n_chunks > total:
                    warnings.warn(
                        f"Number of chunks {n_chunks} is larger than the total number of chunks: {total}+1. "
                        "Some chunks will be empty."
                    )

                chunk_size = total / n_chunks
                parts = [x for x in range(total) if x >= (i_chunk - 1) * chunk_size and x < i_chunk * chunk_size]

        parts = [int(_) for _ in parts]
        LOG.info(f"Running parts: {parts}")
        if not parts:
            warnings.warn(f"Nothing to do for chunk {i_chunk}/{n_chunks}.")

        self.parts = parts

    def __call__(self, i):
        if self.parts is None:
            return True
        return i in self.parts


class ViewCacheArray:
    """A class that provides a caching mechanism for writing to a NumPy-like array.

    The is initialized with a NumPy-like array, a shape and a list to reindex the first dimension.
    The array is used to store the final data, while the cache is used to temporarily
    store the data before flushing it to the array.

    The `flush` method copies the contents of the cache to the final array.
    """

    def __init__(self, array, *, shape, indexes):
        assert len(indexes) == shape[0], (len(indexes), shape[0])
        self.array = array
        self.dtype = array.dtype
        self.cache = np.full(shape, np.nan, dtype=self.dtype)
        self.indexes = indexes

    def __setitem__(self, key, value):
        self.cache[key] = value

    def flush(self):
        for i in range(self.cache.shape[0]):
            global_i = self.indexes[i]
            self.array[global_i] = self.cache[i]
