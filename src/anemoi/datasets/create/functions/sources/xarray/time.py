# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import datetime

from anemoi.utils.dates import as_datetime


class Time:

    @classmethod
    def from_coordinates(cls, coordinates):
        time_coordinate = [c for c in coordinates if c.is_time]
        step_coordinate = [c for c in coordinates if c.is_step]
        date_coordinate = [c for c in coordinates if c.is_date]

        if len(date_coordinate) == 0 and len(time_coordinate) == 1 and len(step_coordinate) == 1:
            return ForecastFromValidTimeAndStep(time_coordinate[0], step_coordinate[0])

        if len(date_coordinate) == 0 and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return Analysis(time_coordinate[0])

        if len(date_coordinate) == 0 and len(time_coordinate) == 0 and len(step_coordinate) == 0:
            return Constant()

        if len(date_coordinate) == 1 and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return ForecastFromValidTimeAndBaseTime(date_coordinate[0], time_coordinate[0])

        if len(date_coordinate) == 1 and len(time_coordinate) == 0 and len(step_coordinate) == 1:
            return ForecastFromBaseTimeAndDate(date_coordinate[0], step_coordinate[0])

        raise NotImplementedError(f"{date_coordinate=} {time_coordinate=} {step_coordinate=}")


class Constant(Time):

    def fill_time_metadata(self, coords_values, metadata):
        return None


class Analysis(Time):

    def __init__(self, time_coordinate):
        self.time_coordinate_name = time_coordinate.variable.name

    def fill_time_metadata(self, coords_values, metadata):
        valid_datetime = coords_values[self.time_coordinate_name]

        metadata["date"] = as_datetime(valid_datetime).strftime("%Y%m%d")
        metadata["time"] = as_datetime(valid_datetime).strftime("%H%M")
        metadata["step"] = 0

        return valid_datetime


class ForecastFromValidTimeAndStep(Time):

    def __init__(self, time_coordinate, step_coordinate):
        self.time_coordinate_name = time_coordinate.variable.name
        self.step_coordinate_name = step_coordinate.variable.name

    def fill_time_metadata(self, coords_values, metadata):
        valid_datetime = coords_values[self.time_coordinate_name]
        step = coords_values[self.step_coordinate_name]

        assert isinstance(step, datetime.timedelta)
        base_datetime = valid_datetime - step

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = as_datetime(base_datetime).strftime("%Y%m%d")
        metadata["time"] = as_datetime(base_datetime).strftime("%H%M")
        metadata["step"] = int(hours)
        return valid_datetime


class ForecastFromValidTimeAndBaseTime(Time):

    def __init__(self, date_coordinate, time_coordinate):
        self.date_coordinate.name = date_coordinate.name
        self.time_coordinate.name = time_coordinate.name

    def fill_time_metadata(self, coords_values, metadata):
        valid_datetime = coords_values[self.time_coordinate_name]
        base_datetime = coords_values[self.date_coordinate_name]

        step = valid_datetime - base_datetime

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = as_datetime(base_datetime).strftime("%Y%m%d")
        metadata["time"] = as_datetime(base_datetime).strftime("%H%M")
        metadata["step"] = int(hours)

        return valid_datetime


class ForecastFromBaseTimeAndDate(Time):

    def __init__(self, date_coordinate, step_coordinate):
        self.date_coordinate_name = date_coordinate.name
        self.step_coordinate_name = step_coordinate.name

    def fill_time_metadata(self, coords_values, metadata):

        date = coords_values[self.date_coordinate_name]
        step = coords_values[self.step_coordinate_name]
        assert isinstance(step, datetime.timedelta)

        metadata["date"] = as_datetime(date).strftime("%Y%m%d")
        metadata["time"] = as_datetime(date).strftime("%H%M")

        hours = step.total_seconds() / 3600

        assert int(hours) == hours
        metadata["step"] = int(hours)

        return date + step
