# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import copy
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

        return method(self, index)

    return wrapper


def make_slice_or_index_from_list_or_tuple(indices):
    """Convert a list or tuple of indices to a slice or an index, if possible."""

    if len(indices) < 2:
        return indices

    step = indices[1] - indices[0]

    if step > 0 and all(indices[i] - indices[i - 1] == step for i in range(1, len(indices))):
        return slice(indices[0], indices[-1] + step, step)

    return indices


def get_indices_for_child_datasets_from_combined_axis_index(
    join_axis: int, index_combined: tuple[slice | list[int],], child_datasets_axis_lengths: list[int]
) -> list[tuple[slice, list[int]]]:
    """Given a combined axis index, and the axis along which the child datasets are joined, return the indices for each child dataset."""

    # 1) ascertain the lengths of each child dataset in the combined axis
    # 2) create new indexing indices for each child dataset by correctly adjusting the combined axis index for each child
    # 3) index each child dataset using the new indices
    # 4) concatenate the results in an order that matches the original combined axis index

    cumulative_lengths = np.cumsum(child_datasets_axis_lengths)

    start_indices_child = [0] + cumulative_lengths[:-1].tolist()
    end_indices_child = [s + len_child for s, len_child in zip(start_indices_child, child_datasets_axis_lengths)]

    index_children = []
    for idx_child in range(len(child_datasets_axis_lengths)):
        index_child = list(copy.deepcopy(index_combined))

        # in the join axis, select the indices that map to the child dataset at position i
        index_at_join_axis = index_child[join_axis]

        if isinstance(index_at_join_axis, slice):
            start, stop, step = index_at_join_axis.indices(cumulative_lengths[idx_child])

            # Ensure the slice is within the bounds
            start = max(start, start_indices_child[idx_child])
            stop = min(stop, end_indices_child[idx_child])

            adjusted_start = start - start_indices_child[idx_child]
            adjusted_stop = stop - start_indices_child[idx_child]
            new_index_at_join_axis = slice(adjusted_start, adjusted_stop, step)
            index_child[join_axis] = new_index_at_join_axis

        elif isinstance(index_at_join_axis, list):
            new_index_at_join_aixs = [
                idx
                for idx in index_at_join_axis
                if idx >= start_indices_child[idx_child] and idx < end_indices_child[idx_child]
            ]
            adjusted_index_at_join_axis = [idx - start_indices_child[idx_child] for idx in new_index_at_join_aixs]
            index_child[join_axis] = adjusted_index_at_join_axis
        else:
            ValueError("Index at join axis is not a slice or a list")

        index_child = tuple(index_child)
        index_children.append(index_child)
    return index_children
