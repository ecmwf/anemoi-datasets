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
        if shape == (1,):
            return to_datetime(variable.values[0])
        assert False, (shape, variable.values[:2])

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
    is_date = False
    is_member = False
    is_x = False
    is_y = False

    def __init__(self, variable):
        self.variable = variable
        self.scalar = is_scalar(variable)
        self.kwargs = {}  # Used when creating a new coordinate (reduced method)

    def __len__(self):
        return 1 if self.scalar else len(self.variable)

    def __repr__(self):
        return "%s[name=%s,values=%s,shape=%s]" % (
            self.__class__.__name__,
            self.variable.name,
            self.variable.values if self.scalar else len(self),
            self.variable.shape,
        )

    def reduced(self, i):
        """Create a new coordinate with a single value

        Parameters
        ----------
        i : int
            the index of the value to select

        Returns
        -------
        Coordinate
            the new coordinate
        """
        return self.__class__(
            self.variable.isel({self.variable.dims[0]: i}),
            **self.kwargs,
        )

    def index(self, value):
        """Return the index of the value in the coordinate

        Parameters
        ----------
        value : Any
            The value to search for

        Returns
        -------
        int or None
            The index of the value in the coordinate or None if not found
        """

        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return self._index_single(value)
            else:
                return self._index_multiple(value)
        return self._index_single(value)

    def _index_single(self, value):

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

    def _index_multiple(self, value):

        values = self.variable.values

        # Assume the array is sorted

        index = np.searchsorted(values, value)
        index = index[index < len(values)]

        if np.all(values[index] == value):
            return index

        # If not found, we need to check if the value is in the array

        index = np.where(np.isin(values, value))[0]

        # We could also return incomplete matches
        if len(index) == len(value):
            return index

        return None

    @property
    def name(self):
        return self.variable.name

    def normalise(self, value):
        # Subclasses to format values that will be added to the field metadata
        return value

    @property
    def single_value(self):
        return extract_single_value(self.variable)


class TimeCoordinate(Coordinate):
    is_time = True
    mars_names = ("valid_datetime",)

    def index(self, time):
        return super().index(np.datetime64(time))


class DateCoordinate(Coordinate):
    is_date = True
    mars_names = ("date",)

    def index(self, date):
        return super().index(np.datetime64(date))


class StepCoordinate(Coordinate):
    is_step = True
    mars_names = ("step",)


class LevelCoordinate(Coordinate):
    mars_names = ("level", "levelist")

    def __init__(self, variable, levtype):
        super().__init__(variable)
        self.levtype = levtype
        # kwargs is used when creating a new coordinate (reduced method)
        self.kwargs = {"levtype": levtype}

    def normalise(self, value):
        # Some netcdf have pressue levels in float
        if int(value) == value:
            return int(value)
        return value


class EnsembleCoordinate(Coordinate):
    is_member = True
    mars_names = ("number",)

    def normalise(self, value):
        if int(value) == value:
            return int(value)
        return value


class LongitudeCoordinate(Coordinate):
    is_grid = True
    is_lon = True
    mars_names = ("longitude",)


class LatitudeCoordinate(Coordinate):
    is_grid = True
    is_lat = True
    mars_names = ("latitude",)


class XCoordinate(Coordinate):
    is_grid = True
    is_x = True
    mars_names = ("x",)


class YCoordinate(Coordinate):
    is_grid = True
    is_y = True
    mars_names = ("y",)


class ScalarCoordinate(Coordinate):
    is_grid = False

    @property
    def mars_names(self):
        return (self.variable.name,)


class UnsupportedCoordinate(Coordinate):
    @property
    def mars_names(self):
        return (self.variable.name,)
