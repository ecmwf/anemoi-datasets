# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import logging

from earthkit.data.core.fieldlist import Field
from earthkit.data.core.fieldlist import math

from .coordinates import extract_single_value
from .coordinates import is_scalar
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
        """Create a new XArrayField object.

        Parameters
        ----------
        owner : Variable
            The variable that owns this field.
        selection : XArrayDataArray
            A 2D sub-selection of the variable's underlying array.
            This is actually a nD object, but the first dimensions are always 1.
            The other two dimensions are latitude and longitude.
        """
        super().__init__(owner.array_backend)

        self.owner = owner
        self.selection = selection

        # Copy the metadata from the owner
        self._md = owner._metadata.copy()

        for coord_name, coord_value in self.selection.coords.items():
            if is_scalar(coord_value):
                # Extract the single value from the scalar dimension
                # and store it in the metadata
                coordinate = owner.by_name[coord_name]
                self._md[coord_name] = coordinate.normalise(extract_single_value(coord_value))

        # print(values.ndim, values.shape, selection.dims)
        # By now, the only dimensions should be latitude and longitude
        self._shape = tuple(list(self.selection.shape)[-2:])
        if math.prod(self._shape) != math.prod(self.selection.shape):
            print(self.selection.ndim, self.selection.shape)
            print(self.selection)
            raise ValueError("Invalid shape for selection")

    @property
    def shape(self):
        return self._shape

    def to_numpy(self, flatten=False, dtype=None, index=None):
        if index is not None:
            values = self.selection[index]
        else:
            values = self.selection

        assert dtype is None

        if flatten:
            return values.values.flatten()

        return values  # .reshape(self.shape)

    def _make_metadata(self):
        return XArrayMetadata(self)

    def grid_points(self):
        return self.owner.grid_points()

    @property
    def resolution(self):
        return None

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
        date, time = self.metadata("date", "time")
        assert len(time) == 4, time
        assert len(date) == 8, date
        yyyymmdd = int(date)
        time = int(time) // 100
        return datetime.datetime(yyyymmdd // 10000, yyyymmdd // 100 % 100, yyyymmdd % 100, time)

    def __repr__(self):
        return repr(self._metadata)

    def _values(self):
        # we don't use .values as this will download the data
        return self.selection
