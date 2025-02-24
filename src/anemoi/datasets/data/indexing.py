# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from functools import wraps
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex


def _tuple_with_slices(t: TupleIndex, shape: Shape) -> Tuple[TupleIndex, Tuple[int, ...]]:
    """Replace all integers in a tuple with slices, so we preserve the dimensionality.

    Parameters:
    t (TupleIndex): The tuple index to process.
    shape (Shape): The shape of the array.

    Returns:
    Tuple[TupleIndex, Tuple[int, ...]]: A tuple containing the modified index and the changes.
    """
    result = tuple(slice(i, i + 1) if isinstance(i, int) else i for i in t)
    changes = tuple(j for (j, i) in enumerate(t) if isinstance(i, int))
    result = tuple(slice(*s.indices(shape[i])) for (i, s) in enumerate(result))

    return result, changes


def _extend_shape(index: TupleIndex, shape: Shape) -> TupleIndex:
    """Extend the shape of the index to match the shape of the array.

    Parameters:
    index (TupleIndex): The index to extend.
    shape (Shape): The shape of the array.

    Returns:
    TupleIndex: The extended index.
    """
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


def _index_to_tuple(index: FullIndex, shape: Shape) -> TupleIndex:
    """Convert an index to a tuple index.

    Parameters:
    index (FullIndex): The index to convert.
    shape (Shape): The shape of the array.

    Returns:
    TupleIndex: The converted tuple index.
    """
    if isinstance(index, int):
        return _extend_shape((index,), shape)
    if isinstance(index, slice):
        return _extend_shape((index,), shape)
    if isinstance(index, tuple):
        return _extend_shape(index, shape)
    if index is Ellipsis:
        return _extend_shape((Ellipsis,), shape)
    raise ValueError(f"Invalid index: {index}")


def index_to_slices(index: Union[int, slice, Tuple], shape: Shape) -> Tuple[TupleIndex, Tuple[int, ...]]:
    """Convert an index to a tuple of slices, with the same dimensionality as the shape.

    Parameters:
    index (Union[int, slice, Tuple]): The index to convert.
    shape (Shape): The shape of the array.

    Returns:
    Tuple[TupleIndex, Tuple[int, ...]]: A tuple containing the slices and the changes.
    """
    return _tuple_with_slices(_index_to_tuple(index, shape), shape)


def apply_index_to_slices_changes(result: NDArray[Any], changes: Tuple[int, ...]) -> NDArray[Any]:
    """Apply changes to the result array based on the slices.

    Parameters:
    result (NDArray[Any]): The result array.
    changes (Tuple[int, ...]): The changes to apply.

    Returns:
    NDArray[Any]: The modified result array.
    """
    if changes:
        shape = result.shape
        for i in changes:
            assert shape[i] == 1, (i, changes, shape)
        result = np.squeeze(result, axis=changes)
    return result


def update_tuple(t: Tuple, index: int, value: Any) -> Tuple[Tuple, Any]:
    """Replace the elements of a tuple at the given index with a new value.

    Parameters:
    tp (Tuple): The original tuple.
    index (int): The index to update.
    value (Any): The new value.

    Returns:
    Tuple[Tuple, Any]: The updated tuple and the previous value.
    """
    t = list(t)
    prev = t[index]
    t[index] = value
    return tuple(t), prev


def length_to_slices(index: slice, lengths: List[int]) -> List[Union[slice, None]]:
    """Convert an index to a list of slices, given the lengths of the dimensions.

    Parameters:
    index (slice): The index to convert.
    lengths (List[int]): The lengths of the dimensions.

    Returns:
    List[Union[slice, None]]: A list of slices.
    """
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


def _as_tuples(index: Tuple) -> Tuple:
    """Convert elements of the index to tuples if they are lists or arrays.

    Parameters:
    index (Tuple): The index to convert.

    Returns:
    Tuple: The converted index.
    """

    def _(i: Any) -> Any:
        if hasattr(i, "tolist"):
            # NumPy arrays, TensorFlow tensors, etc.
            i = i.tolist()
            assert not isinstance(i[0], bool), "Mask not supported"
            return tuple(i)

        if isinstance(i, list):
            return tuple(i)

        return i

    return tuple(_(i) for i in index)


def expand_list_indexing(method: Callable[..., NDArray[Any]]) -> Callable[..., NDArray[Any]]:
    """Allows to use slices, lists, and tuples to select data from the dataset.
    Zarr does not support indexing with lists/arrays directly,
    so we need to implement it ourselves.

    Parameters:
    method (Callable[..., NDArray[Any]]): The method to wrap.

    Returns:
    Callable[..., NDArray[Any]]: The wrapped method.
    """

    @wraps(method)
    def wrapper(self: Any, index: FullIndex) -> NDArray[Any]:
        if not isinstance(index, tuple):
            return method(self, index)

        if not any(isinstance(i, (list, tuple)) for i in index):
            return method(self, index)

        which: List[int] = []
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


def make_slice_or_index_from_list_or_tuple(indices: List[int]) -> Union[List[int], slice]:
    """Convert a list or tuple of indices to a slice or an index, if possible.

    Parameters:
    indices (List[int]): The list or tuple of indices.

    Returns:
    Union[List[int], slice]: The slice or index.
    """
    if len(indices) < 2:
        return indices

    step = indices[1] - indices[0]

    if step > 0 and all(indices[i] - indices[i - 1] == step for i in range(1, len(indices))):
        return slice(indices[0], indices[-1] + step, step)

    return indices
