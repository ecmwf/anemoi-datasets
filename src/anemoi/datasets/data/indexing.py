# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import wraps

import numpy as np


def _tuple_with_slices(t, shape):
    """Replace all integers in a tuple with slices, so we preserve the dimensionality."""

    result = tuple(slice(i, i + 1) if isinstance(i, int) else i for i in t)
    changes = tuple(j for (j, i) in enumerate(t) if isinstance(i, int))
    result = tuple(slice(*s.indices(shape[i])) for (i, s) in enumerate(result))

    return result, changes


def _extend_shape(index, shape):
    if Ellipsis in index:
        if index.count(Ellipsis) > 1:
            raise IndexError("Only one Ellipsis is allowed")
        ellipsis_index = index.index(Ellipsis)
        index = list(index)
        index[ellipsis_index] = slice(None)
        while len(index) < len(shape):
            index.insert(ellipsis_index, slice(None))
        index = tuple(index)

    while len(index) < len(shape):
        index = index + (slice(None),)

    return index


def _index_to_tuple(index, shape):
    if isinstance(index, int):
        return _extend_shape((index,), shape)
    if isinstance(index, slice):
        return _extend_shape((index,), shape)
    if isinstance(index, tuple):
        return _extend_shape(index, shape)
    if index is Ellipsis:
        return _extend_shape((Ellipsis,), shape)
    raise ValueError(f"Invalid index: {index}")


def index_to_slices(index, shape):
    """Convert an index to a tuple of slices, with the same dimensionality as the shape."""
    return _tuple_with_slices(_index_to_tuple(index, shape), shape)


def apply_index_to_slices_changes(result, changes):
    if changes:
        shape = result.shape
        for i in changes:
            assert shape[i] == 1, (i, changes, shape)
        result = np.squeeze(result, axis=changes)
    return result


def update_tuple(t, index, value):
    """Replace the elements of a tuple at the given index with a new value."""
    t = list(t)
    prev = t[index]
    t[index] = value
    return tuple(t), prev


def length_to_slices(index, lengths):
    """Convert an index to a list of slices, given the lengths of the dimensions."""
    total = sum(lengths)
    start, stop, step = index.indices(total)

    result = []

    pos = 0
    for length in lengths:
        end = pos + length

        b = max(pos, start)
        e = min(end, stop)

        p = None
        if b <= e:
            if (b - start) % step != 0:
                b = b + step - (b - start) % step
            b -= pos
            e -= pos

            if 0 <= b < e:
                p = slice(b, e, step)

        result.append(p)

        pos = end

    return result


def _as_tuples(index):
    def _(i):
        if hasattr(i, "tolist"):
            # NumPy arrays, TensorFlow tensors, etc.
            i = i.tolist()
            assert not isinstance(i[0], bool), "Mask not supported"
            return tuple(i)

        if isinstance(i, list):
            return tuple(i)

        return i

    return tuple(_(i) for i in index)


def expand_list_indexing(method):
    """Allows to use slices, lists, and tuples to select data from the dataset. Zarr does not support indexing with lists/arrays directly, so we need to implement it ourselves."""

    @wraps(method)
    def wrapper(self, index):
        if not isinstance(index, tuple):
            return method(self, index)

        if not any(isinstance(i, (list, tuple)) for i in index):
            return method(self, index)

        which = []
        for i, idx in enumerate(index):
            if isinstance(idx, (list, tuple)):
                which.append(i)

        assert which, "No list index found"

        if len(which) > 1:
            raise IndexError("Only one list index is allowed")

        which = which[0]
        index = _as_tuples(index)
        result = []
        for i in index[which]:
            index, _ = update_tuple(index, which, slice(i, i + 1))
            result.append(method(self, index))

        return np.concatenate(result, axis=which)

    return wrapper


def make_slice_or_index_from_list_or_tuple(indices):
    """Convert a list or tuple of indices to a slice or an index, if possible."""

    if len(indices) < 2:
        return indices

    step = indices[1] - indices[0]

    if step > 0 and all(indices[i] - indices[i - 1] == step for i in range(1, len(indices))):
        return slice(indices[0], indices[-1] + step, step)

    return indices
