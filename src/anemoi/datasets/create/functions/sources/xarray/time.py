# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import datetime


class Time:
    @classmethod
    def from_coordinates(cls, coordinates):
        time_coordinate = [c for c in coordinates if c.is_time]
        step_coordinate = [c for c in coordinates if c.is_step]
        date_coordinate = [c for c in coordinates if c.is_date]

        if len(date_coordinate) == 0 and len(time_coordinate) == 1 and len(step_coordinate) == 1:
            return ForecasstFromValidTimeAndStep(step_coordinate[0])

        if len(date_coordinate) == 0 and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return Analysis()

        if len(date_coordinate) == 0 and len(time_coordinate) == 0 and len(step_coordinate) == 0:
            return Constant()

        if len(date_coordinate) == 1 and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return ForecastFromValidTimeAndBaseTime(date_coordinate[0])

        if len(date_coordinate) == 1 and len(time_coordinate) == 0 and len(step_coordinate) == 1:
            return ForecastFromBaseTimeAndDate(date_coordinate[0], step_coordinate[0])

        raise NotImplementedError(f"{date_coordinate=} {time_coordinate=} {step_coordinate=}")


class Constant(Time):

    def fill_time_metadata(self, time, metadata):
        metadata["date"] = time.strftime("%Y%m%d")
        metadata["time"] = time.strftime("%H%M")
        metadata["step"] = 0


class Analysis(Time):

    def fill_time_metadata(self, time, metadata):
        metadata["date"] = time.strftime("%Y%m%d")
        metadata["time"] = time.strftime("%H%M")
        metadata["step"] = 0


class ForecasstFromValidTimeAndStep(Time):
    def __init__(self, step_coordinate):
        self.step_name = step_coordinate.variable.name

    def fill_time_metadata(self, time, metadata):
        step = metadata.pop(self.step_name)
        assert isinstance(step, datetime.timedelta)
        base = time - step

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = base.strftime("%Y%m%d")
        metadata["time"] = base.strftime("%H%M")
        metadata["step"] = int(hours)


class ForecastFromValidTimeAndBaseTime(Time):
    def __init__(self, date_coordinate):
        self.date_coordinate = date_coordinate

    def fill_time_metadata(self, time, metadata):

        step = time - self.date_coordinate

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = self.date_coordinate.single_value.strftime("%Y%m%d")
        metadata["time"] = self.date_coordinate.single_value.strftime("%H%M")
        metadata["step"] = int(hours)


class ForecastFromBaseTimeAndDate(Time):
    def __init__(self, date_coordinate, step_coordinate):
        self.date_coordinate = date_coordinate
        self.step_coordinate = step_coordinate

    def fill_time_metadata(self, time, metadata):
        metadata["date"] = time.strftime("%Y%m%d")
        metadata["time"] = time.strftime("%H%M")
        hours = metadata[self.step_coordinate.name].total_seconds() / 3600
        assert int(hours) == hours
        metadata["step"] = int(hours)
