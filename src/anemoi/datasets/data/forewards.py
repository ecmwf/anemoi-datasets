# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np

from .dataset import Dataset
from .debug import debug_indexing
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import length_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Forwards(Dataset):
    def __init__(self, forward):
        self.forward = forward

    def __len__(self):
        return len(self.forward)

    def __getitem__(self, n):
        return self.forward[n]

    @property
    def dates(self):
        return self.forward.dates

    @property
    def resolution(self):
        return self.forward.resolution

    @property
    def field_shape(self):
        return self.forward.field_shape

    @property
    def frequency(self):
        return self.forward.frequency

    @property
    def latitudes(self):
        return self.forward.latitudes

    @property
    def longitudes(self):
        return self.forward.longitudes

    @property
    def name_to_index(self):
        return self.forward.name_to_index

    @property
    def variables(self):
        return self.forward.variables

    @property
    def statistics(self):
        return self.forward.statistics

    @property
    def shape(self):
        return self.forward.shape

    @property
    def dtype(self):
        return self.forward.dtype

    @property
    def missing(self):
        return self.forward.missing

    @property
    def grids(self):
        return self.forward.grids

    def metadata_specific(self, **kwargs):
        return super().metadata_specific(
            forward=self.forward.metadata_specific(),
            **kwargs,
        )

    def source(self, index):
        return self.forward.source(index)


class Combined(Forwards):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

        # Forward most properties to the first dataset
        super().__init__(datasets[0])

    def check_same_resolution(self, d1, d2):
        if d1.resolution != d2.resolution:
            raise ValueError(f"Incompatible resolutions: {d1.resolution} and {d2.resolution} ({d1} {d2})")

    def check_same_frequency(self, d1, d2):
        if d1.frequency != d2.frequency:
            raise ValueError(f"Incompatible frequencies: {d1.frequency} and {d2.frequency} ({d1} {d2})")

    def check_same_grid(self, d1, d2):
        if (d1.latitudes != d2.latitudes).any() or (d1.longitudes != d2.longitudes).any():
            raise ValueError(f"Incompatible grid ({d1} {d2})")

    def check_same_shape(self, d1, d2):
        if d1.shape[1:] != d2.shape[1:]:
            raise ValueError(f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})")

        if d1.variables != d2.variables:
            raise ValueError(f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})")

    def check_same_sub_shapes(self, d1, d2, drop_axis):
        shape1 = d1.sub_shape(drop_axis)
        shape2 = d2.sub_shape(drop_axis)

        if shape1 != shape2:
            raise ValueError(f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})")

    def check_same_variables(self, d1, d2):
        if d1.variables != d2.variables:
            raise ValueError(f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})")

    def check_same_lengths(self, d1, d2):
        if d1._len != d2._len:
            raise ValueError(f"Incompatible lengths: {d1._len} and {d2._len}")

    def check_same_dates(self, d1, d2):
        self.check_same_frequency(d1, d2)

        if d1.dates[0] != d2.dates[0]:
            raise ValueError(f"Incompatible start dates: {d1.dates[0]} and {d2.dates[0]} ({d1} {d2})")

        if d1.dates[-1] != d2.dates[-1]:
            raise ValueError(f"Incompatible end dates: {d1.dates[-1]} and {d2.dates[-1]} ({d1} {d2})")

    def check_compatibility(self, d1, d2):
        # These are the default checks
        # Derived classes should turn individual checks off if they are not needed
        self.check_same_resolution(d1, d2)
        self.check_same_frequency(d1, d2)
        self.check_same_grid(d1, d2)
        self.check_same_lengths(d1, d2)
        self.check_same_variables(d1, d2)
        self.check_same_dates(d1, d2)

    def provenance(self):
        return [d.provenance() for d in self.datasets]

    def __repr__(self):
        lst = ", ".join(repr(d) for d in self.datasets)
        return f"{self.__class__.__name__}({lst})"

    def metadata_specific(self, **kwargs):
        # We need to skip the forward superclass
        # TODO: revisit this
        return Dataset.metadata_specific(
            self,
            datasets=[d.metadata_specific() for d in self.datasets],
            **kwargs,
        )

    @cached_property
    def missing(self):
        offset = 0
        result = set()
        for d in self.datasets:
            result.update(offset + m for m in d.missing)
            offset += len(d)
        return result


class GivenAxis(Combined):
    """Given a given axis, combine the datasets along that axis."""

    def __init__(self, datasets, axis):
        self.axis = axis
        super().__init__(datasets)

        assert axis > 0 and axis < len(self.datasets[0].shape), (
            axis,
            self.datasets[0].shape,
        )

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=self.axis)

    @cached_property
    def shape(self):
        shapes = [d.shape for d in self.datasets]
        before = shapes[0][: self.axis]
        after = shapes[0][self.axis + 1 :]
        result = before + (sum(s[self.axis] for s in shapes),) + after
        assert False not in result, result
        return result

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        lengths = [d.shape[self.axis] for d in self.datasets]
        slices = length_to_slices(index[self.axis], lengths)
        result = [d[update_tuple(index, self.axis, i)[0]] for (d, i) in zip(self.datasets, slices) if i is not None]
        result = np.concatenate(result, axis=self.axis)
        return apply_index_to_slices_changes(result, changes)

    @debug_indexing
    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        return np.concatenate([d[n] for d in self.datasets], axis=self.axis - 1)
