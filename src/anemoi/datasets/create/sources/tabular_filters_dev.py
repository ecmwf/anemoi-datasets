# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import sys

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry
from earthkit.data.utils.dates import to_datetime

from anemoi.datasets.usage.grids import cropping_mask

LOG = logging.getLogger(__name__)


class OverloadDispatcher:
    def __init__(self, name):
        self.name = name
        self.registry = {}

    def register(self, arg_type, func):
        self.registry[arg_type] = func

    def __call__(self, instance, *args, **kwargs):
        # We assume the first argument (after self) is the one to check
        if not args:
            raise TypeError(f"{self.name}() requires at least one argument")

        arg = args[0]
        arg_type = type(arg)

        for registered_type, func in self.registry.items():
            if issubclass(arg_type, registered_type):
                return func(instance, *args, **kwargs)

        raise TypeError(f"No match found for type {arg_type.__name__} in {self.name}()")

    def __get__(self, instance, owner):
        # This makes the dispatcher work as a method (binding 'self')
        if instance is None:
            return self
        return lambda *args, **kwargs: self.__call__(instance, *args, **kwargs)


def _create_overload(target_type, func):
    func_name = func.__name__
    # Get the namespace where the function is being defined (the class body)
    frame = sys._getframe(2)
    locals_dict = frame.f_locals

    # Get the existing dispatcher or create a new one
    dispatcher = locals_dict.get(func_name)
    if not isinstance(dispatcher, OverloadDispatcher):
        dispatcher = OverloadDispatcher(func_name)

    dispatcher.register(target_type, func)
    return dispatcher


def expect_tabular(func):
    import pandas as pd

    return _create_overload(pd.DataFrame, func)


def expect_gridded(func):
    import earthkit.data as ekd

    return _create_overload(ekd.FieldList, func)


@filter_registry.register("crop")
class Crop(Filter):

    def __init__(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
    ):

        self.north = north
        self.south = south
        self.east = east % 360  # Ensure east is within [0, 360)
        self.west = west % 360  # Ensure west is within [0, 360)

    @expect_tabular
    def forward(self, frame):
        LOG.info(f"Cropping to N:{self.north}, S:{self.south}, E:{self.east}, W:{self.west}")
        LOG.info(f"Parameter type: {type(frame)}")

        # Just a test, does not support

        # Assume frame is a pandas DataFrame with 'latitude' and 'longitude' columns
        # Code to be checked

        # Handle longitude wrapping (date line)
        if self.east < self.west:
            # Crop region crosses the date line
            lon_mask = (frame["longitude"] >= self.west) | (frame["longitude"] <= self.east)
        else:
            lon_mask = (frame["longitude"] >= self.west) & (frame["longitude"] <= self.east)

        lat_mask = (frame["latitude"] >= self.south) & (frame["latitude"] <= self.north)

        frame = frame[lat_mask & lon_mask]

        return frame

    @expect_gridded
    def forward(self, fields):  # noqa: F811
        LOG.info(f"Cropping to N:{self.north}, S:{self.south}, E:{self.east}, W:{self.west}")
        LOG.info(f"Parameter type: {type(fields)}")

        mask = None
        latitudes = None
        longitudes = None

        result = []

        for field in fields:
            if mask is None:  # Assume all fields have the same grid
                latitudes, longitudes = field.grid_points()

                mask = cropping_mask(
                    latitudes,
                    longitudes,
                    north=self.north,
                    south=self.south,
                    east=self.east,
                    west=self.west,
                )

                latitudes = latitudes[mask]
                longitudes = longitudes[mask]

            array = field.to_numpy()
            cropped_array = array[mask]
            result.append(
                new_field_from_numpy(
                    cropped_array,
                    template=new_field_from_latitudes_longitudes(field, latitudes, longitudes),
                )
            )

        return new_fieldlist_from_list(result)


@filter_registry.register("tabularise")
class Tabularise(Filter):

    def __init__(
        self,
        param: list[str] = None,
    ):
        self.param = param if param is not None else None

    @expect_gridded
    def forward(self, fields):
        frames = {}
        assert len(fields)
        for field in fields:

            param = field.metadata("param")
            if self.param is not None and param not in self.param:
                continue

            date = to_datetime(field.metadata("valid_datetime"))

            if param not in frames:
                frames[param] = pd.DataFrame(columns=["date", "latitude", "longitude", param])

            latitudes, longitudes, values = field.data()

            df = pd.DataFrame(
                {
                    "date": pd.Series([date] * len(latitudes), dtype="datetime64[ns]"),
                    "latitude": pd.Series(latitudes, dtype="float32"),
                    "longitude": pd.Series(longitudes, dtype="float32"),
                    param: pd.Series(values, dtype="float32"),
                }
            )

            frames[param] = pd.concat([frames[param], df], ignore_index=True)

        # Merge all frames with matching date, latitude, and longitude
        result = []
        for param, df in frames.items():
            df = df.set_index(["date", "latitude", "longitude"])
            print(df)
            result.append(df)

        if result:
            result = pd.concat(result, axis=1).reset_index()
        else:
            result = pd.DataFrame(
                {
                    "date": pd.Series([], dtype="datetime64[ns]"),
                    "latitude": pd.Series([], dtype="float32"),
                    "longitude": pd.Series([], dtype="float32"),
                    **{
                        param: pd.Series([], dtype="float32")
                        for param in (self.param if self.param is not None else [])
                    },
                }
            )

        if self.param is not None:
            # Use user's specified parameters order
            result = result[["date", "latitude", "longitude"] + self.param]

        return result


@filter_registry.register("griddify")
class Griddify(Filter):

    def __init__(
        self,
        template: str,
        max_distance_km: float = 200.0,
    ):
        from anemoi.utils.grids import latlon_to_xyz
        from scipy.spatial import KDTree

        self.template = ekd.from_source("file", template)[0]
        atitudes, longitudes = self.template.grid_points()

        xyz = latlon_to_xyz(atitudes, longitudes)
        self.tree = KDTree(np.array(xyz).transpose())
        self.length = len(atitudes)

        self.max_distance = max_distance_km / 6371.0  # Convert from km to radians

    @expect_tabular
    def forward(self, frame):
        from anemoi.utils.grids import latlon_to_xyz

        result = []
        value_columns = [col for col in frame.columns if col not in ("date", "latitude", "longitude")]

        for col in value_columns:
            for date, group in frame.groupby("date"):
                # Only keep relevant columns for this variable and date
                df = group[["date", "latitude", "longitude", col]]

                xyz = latlon_to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
                points = np.array(xyz).transpose()

                distances, indices = self.tree.query(points)
                gridded_values = np.full(self.length, np.nan, dtype="float32")
                for i, (distance, idx) in enumerate(zip(distances, indices)):
                    if distance < self.max_distance:
                        gridded_values[idx] = df.iloc[i][col]

                result.append(
                    new_field_with_valid_datetime(
                        new_field_from_numpy(gridded_values, template=self.template, valid_datetime=date, param=col),
                        valid_datetime=date,
                    )
                )

        return new_fieldlist_from_list(result)
