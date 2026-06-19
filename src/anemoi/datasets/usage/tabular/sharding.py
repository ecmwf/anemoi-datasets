# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    from anemoi.datasets.usage.dataset import Dataset

LOG = logging.getLogger(__name__)


def _validate_count(count: int) -> None:
    if not (isinstance(count, int) and not isinstance(count, bool) and count >= 1):
        raise ValueError(f"sharding must be a positive integer, got {count!r}")


class ShardedTabular(Sequence):
    """A list-like collection of the shards of a tabular dataset.

    Returned by ``open_dataset(path, sharding=N)``. It behaves like a list of
    ``N`` shard datasets — ``len(...)``, ``shards[i]``, and iteration all yield
    the individual shard datasets — and additionally exposes the aggregate
    sizing information needed to reassemble the shards into the full dataset.

    The sizing attributes (:attr:`unsharded_sizes`, :attr:`shard_sizes`,
    :attr:`total_size`) are identical to those of each individual shard.
    """

    def __init__(self, shards: "list[Dataset]") -> None:
        self._shards = list(shards)

    def __len__(self) -> int:
        return len(self._shards)

    def __getitem__(self, index: "int | slice") -> "Dataset | list[Dataset]":
        return self._shards[index]

    def __iter__(self) -> "Iterator[Dataset]":
        return iter(self._shards)

    @property
    def shards(self) -> "list[Dataset]":
        """The shard datasets, as a list."""
        return list(self._shards)

    @property
    def num_shards(self) -> int:
        """The number of shards."""
        return len(self._shards)

    @property
    def unsharded_sizes(self) -> NDArray:
        """Per-window row counts of the unsharded dataset."""
        return self._shards[0].unsharded_sizes

    @property
    def shard_sizes(self) -> NDArray:
        """Per-shard, per-window row counts, shaped ``(num_shards, num_windows)``."""
        return self._shards[0].shard_sizes

    @property
    def total_size(self) -> int:
        """Total row count of the unsharded dataset across all windows."""
        return self._shards[0].total_size

    def __repr__(self) -> str:
        return f"ShardedTabular(num_shards={self.num_shards}, total_size={self.total_size})"


def shard_dataset(
    ds: "Dataset",
    count: int,
    wrap: "Callable[[Dataset], Dataset] | None" = None,
) -> ShardedTabular:
    """Split a tabular dataset into ``count`` shards.

    Returns a :class:`ShardedTabular`. Each shard accepts the same methods and
    attributes as ``ds`` but every window yields only its share of the rows (see
    ``Dataset.shard_view``). The window count, dates, frequency and variables
    are identical across shards. Each shard's :attr:`Dataset.other_shards` is
    populated with its siblings.

    Parameters
    ----------
    ds : Dataset
        The tabular dataset to shard.
    count : int
        The number of shards.
    wrap : callable, optional
        Applied to each shard before sibling linking (e.g. to wrap in ``Trace``).

    Returns
    -------
    ShardedTabular
        The collection of shard datasets.
    """
    _validate_count(count)

    shards = [ds.shard_view(index, count) for index in range(count)]
    if wrap is not None:
        shards = [wrap(shard) for shard in shards]

    for shard in shards:
        shard._other_shards = [other for other in shards if other is not shard]

    return ShardedTabular(shards)


def single_shard(
    ds: "Dataset",
    count: int,
    index: int,
    wrap: "Callable[[Dataset], Dataset] | None" = None,
) -> "Dataset":
    """Build a single shard ``index`` of ``count`` from a tabular dataset.

    Equivalent to ``shard_dataset(ds, count)[index]`` but builds only the one
    shard. Its :attr:`Dataset.other_shards` is ``None`` (the siblings are not
    materialised), while :attr:`Dataset.shard_sizes` and
    :attr:`Dataset.total_size` still describe the whole partition, so a worker
    holding only this shard can compute where its rows belong.

    Parameters
    ----------
    ds : Dataset
        The tabular dataset to shard.
    count : int
        The total number of shards.
    index : int
        The shard index, in ``[0, count)``.
    wrap : callable, optional
        Applied to the shard (e.g. to wrap in ``Trace``).

    Returns
    -------
    Dataset
        The single shard dataset.
    """
    _validate_count(count)
    if not (isinstance(index, int) and not isinstance(index, bool) and 0 <= index < count):
        raise ValueError(f"shard index must be in [0, {count}), got {index!r}")

    shard = ds.shard_view(index, count)
    if wrap is not None:
        shard = wrap(shard)
    shard._other_shards = None
    return shard
