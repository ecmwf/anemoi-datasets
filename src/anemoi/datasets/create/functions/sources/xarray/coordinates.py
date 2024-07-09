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

import numpy as np
from earthkit.data.utils.dates import to_datetime

LOG = logging.getLogger(__name__)


def is_scalar(variable):
    shape = variable.shape
    if shape == (1,):
        return True
    if len(shape) == 0:
        return True
    return False


def extract_single_value(variable):
    shape = variable.shape
    if np.issubdtype(variable.values.dtype, np.datetime64):
        if len(shape) == 0:
            return to_datetime(variable.values)  # Convert to python datetime
        assert False, (shape, variable.values)

    if np.issubdtype(variable.values.dtype, np.timedelta64):
        if len(shape) == 0:
            # Convert to python timedelta64
            return datetime.timedelta(seconds=variable.values.astype("timedelta64[s]").astype(int).item())
        assert False, (shape, variable.values)

    if shape == (1,):
        return variable.values[0]

    if len(shape) == 0:
        return variable.values.item()

    assert False, (shape, variable.values)


class Coordinate:
    is_grid = False
    is_dim = True
    is_lat = False
    is_lon = False
    is_time = False
    is_step = False

    def __init__(self, variable):
        self.variable = variable
        self.scalar = is_scalar(variable)
        self.kwargs = {}
        # print(self)

    def __len__(self):
        return 1 if self.scalar else len(self.variable)

    def __repr__(self):
        return "%s[name=%s,values=%s]" % (
            self.__class__.__name__,
            self.variable.name,
            self.variable.values if self.scalar else len(self),
        )

    def singleton(self, i):
        return self.__class__(self.variable.isel({self.variable.dims[0]: i}), **self.kwargs)

    def index(self, value):

        values = self.variable.values

        # Assume the array is sorted
        index = np.searchsorted(values, value)
        if index < len(values) and values[index] == value:
            return index

        # If not found, we need to check if the value is in the array

        index = np.where(values == value)[0]
        if len(index) > 0:
            return index[0]

        return None

    @property
    def name(self):
        return self.variable.name

    def update_metadata(self, metadata):
        pass


class TimeCoordinate(Coordinate):
    is_time = True

    def index(self, time):
        return super().index(np.datetime64(time))

    def update_metadata(self, metadata):
        if self.scalar:
            assert False, self.variable


class StepCoordinate(Coordinate):
    is_step = True


class LevelCoordinate(Coordinate):

    def __init__(self, variable, levtype):
        super().__init__(variable)
        self.levtype = levtype
        self.kwargs = {"levtype": levtype}

    def update_metadata(self, metadata):
        metadata.update(self.kwargs)


class EnsembleCoordinate(Coordinate):
    pass


class OtherCoordinate(Coordinate):
    pass


class LongitudeCoordinate(Coordinate):
    is_grid = True
    is_lon = True


class LatitudeCoordinate(Coordinate):
    is_grid = True
    is_lat = True


class XCoordinate(Coordinate):
    is_grid = True


class YCoordinate(Coordinate):
    is_grid = True


class ScalarCoordinate(Coordinate):
    pass
