# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from numpy.typing import NDArray

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import Node
from .debug import Source
from .debug import debug_indexing
from .forwards import Combined
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Join(Combined):
    """Join the datasets along the variables axis."""

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Check the compatibility of two datasets.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=1)

    def check_same_variables(self, d1: Dataset, d2: Dataset) -> None:
        """Check if the datasets have the same variables.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        # Turned off because we are joining along the variables axis
        pass

    def __len__(self) -> int:
        """Get the length of the joined dataset.

        Returns
        -------
        int
            The length of the joined dataset.
        """
        return len(self.datasets[0])

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get the data for a tuple index.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The data for the tuple index.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 1, slice(None))

        # TODO: optimize if index does not access all datasets, so we don't load chunks we don't need
        result = [d[index] for d in self.datasets]

        result = np.concatenate(result, axis=1)
        return apply_index_to_slices_changes(result[:, previous], changes)

    @debug_indexing
    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Get the data for a slice.

        Parameters
        ----------
        s : slice
            The slice to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The data for the slice.
        """
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get the data at the specified index.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The data at the specified index.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        return np.concatenate([d[n] for d in self.datasets])

    @cached_property
    def shape(self) -> Shape:
        """Get the shape of the joined dataset."""
        cols = sum(d.shape[1] for d in self.datasets)
        return (len(self), cols) + self.datasets[0].shape[2:]

    def _overlay(self) -> Dataset:
        """Overlay the datasets.

        Returns
        -------
        Dataset
            The overlaid dataset.
        """
        indices = {}
        i = 0
        for d in self.datasets:
            for v in d.variables:
                indices[v] = i
                i += 1

        if len(indices) == i:
            # No overlay
            return self

        variables = [v[1:-1] for v in self.variables if v[0] == "(" and v[-1] == ")"]
        indices = list(indices.values())

        i = 0
        for d in self.datasets:
            ok = False
            for v in d.variables:
                if i in indices:
                    ok = True
                i += 1
            if not ok:
                LOG.warning("Dataset %r completely overridden.", d)

        from .select import Select

        return Select(self, indices, {"overlay": variables})

    @cached_property
    def variables(self) -> List[str]:
        """Get the variables of the joined dataset."""
        seen = set()
        result: List[str] = []
        for d in reversed(self.datasets):
            for v in reversed(d.variables):
                while v in seen:
                    v = f"({v})"
                seen.add(v)
                result.insert(0, v)

        return result

    @property
    def variables_metadata(self) -> Dict[str, Any]:
        """Get the metadata of the variables."""
        result = {}
        variables = [v for v in self.variables if not (v.startswith("(") and v.endswith(")"))]

        for d in self.datasets:
            md = d.variables_metadata
            for v in variables:
                if v in md:
                    result[v] = md[v]

        if len(result) != len(variables):
            LOG.error("Some variables are missing metadata.")
            for v in variables:
                if v not in result:
                    LOG.error("Missing metadata for %r.", v)

        return result

    @cached_property
    def name_to_index(self) -> Dict[str, int]:
        """Get the mapping of variable names to indices."""
        return {k: i for i, k in enumerate(self.variables)}

    @property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Get the statistics of the joined dataset."""
        return {
            k: np.concatenate([d.statistics[k] for d in self.datasets], axis=0) for k in self.datasets[0].statistics
        }

    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        """Get the statistics tendencies of the joined dataset.

        Parameters
        ----------
        delta : Optional[datetime.timedelta]
            The time delta for the tendencies.

        Returns
        -------
        Dict[str, NDArray[Any]]
            The statistics tendencies of the joined dataset.
        """
        if delta is None:
            delta = self.frequency
        return {
            k: np.concatenate([d.statistics_tendencies(delta)[k] for d in self.datasets], axis=0)
            for k in self.datasets[0].statistics_tendencies(delta)
        }

    def source(self, index: int) -> Source:
        """Get the source of the data at the specified index.

        Parameters
        ----------
        index : int
            The index to retrieve the source from.

        Returns
        -------
        Source
            The source of the data at the specified index.
        """
        i = index
        for dataset in self.datasets:
            if i < dataset.shape[1]:
                return Source(self, index, dataset.source(i))
            i -= dataset.shape[1]
        assert False

    @cached_property
    def missing(self) -> Set[int]:
        """Get the missing data indices."""
        result: Set[int] = set()
        for d in self.datasets:
            result = result | d.missing
        return result

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [d.tree() for d in self.datasets])

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns
        -------
        dict[str, Any]
            The metadata specific to the forwards subclass.
        """
        return {}


def join_factory(args: tuple, kwargs: dict) -> Dataset:
    """Create a joined dataset.

    Parameters
    ----------
    args : tuple
        The positional arguments.
    kwargs : dict
        The keyword arguments.

    Returns
    -------
    Dataset
        The joined dataset.
    """
    datasets = kwargs.pop("join")
    assert isinstance(datasets, (list, tuple))
    assert len(args) == 0

    assert isinstance(datasets, (list, tuple))

    datasets = [_open(e) for e in datasets]

    if len(datasets) == 1:
        return datasets[0]._subset(**kwargs)

    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Join(datasets)._overlay()._subset(**kwargs)
