# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from earthkit.data.core.fieldlist import Field

from .coordinates import _extract_single_value
from .coordinates import _is_scalar
from .metadata import XArrayMetadata

LOG = logging.getLogger(__name__)


class EmptyFieldList:
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)

    def __repr__(self) -> str:
        return "EmptyFieldList()"


class XArrayField(Field):

    def __init__(self, owner, selection):
        super().__init__(owner.array_backend)

        self.owner = owner
        self.selection = selection
        self._md = owner._metadata.copy()

        for coord_name, coord_value in self.selection.coords.items():
            if coord_name in selection.dims:
                continue

            if _is_scalar(coord_value):
                self._md[coord_name] = _extract_single_value(coord_value)

    def to_numpy(self, flatten=False, dtype=None):
        assert dtype is None
        if flatten:
            return self.selection.values.flatten()
        return self.selection.values

    def _make_metadata(self):
        return XArrayMetadata(self)

    def grid_points(self):
        return self.owner.grid_points()

    @property
    def resolution(self):
        return None

    @property
    def shape(self):
        return self.selection.shape

    @property
    def grid_mapping(self):
        return self.owner.grid_mapping

    @property
    def latitudes(self):
        return self.owner.latitudes

    @property
    def longitudes(self):
        return self.owner.longitudes

    @property
    def forecast_reference_time(self):
        return self.owner.forecast_reference_time

    def __repr__(self):
        return repr(self._metadata)
