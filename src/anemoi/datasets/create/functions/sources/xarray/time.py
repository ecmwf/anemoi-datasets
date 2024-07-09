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
    def from_coordinates(cls, forecast_reference_time, coordinates):
        time_coordinate = [c for c in coordinates if c.is_time]
        step_coordinate = [c for c in coordinates if c.is_step]

        if forecast_reference_time is None and len(time_coordinate) == 1 and len(step_coordinate) == 1:
            return ForecasstFromValidTimeAndStep(time_coordinate[0], step_coordinate[0])

        if forecast_reference_time is None and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return Analysis()

        if forecast_reference_time is None and len(time_coordinate) == 0 and len(step_coordinate) == 0:
            return Constant()

        if forecast_reference_time is not None and len(time_coordinate) == 1 and len(step_coordinate) == 0:
            return ForecastFromValidTimeAndBaseTime(forecast_reference_time)

        raise NotImplementedError(f"{forecast_reference_time=} {time_coordinate=} {step_coordinate=}")


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
    def __init__(self, time_coordinate, step_coordinate):
        self.time_coordinate = time_coordinate
        self.step_coordinate = step_coordinate

    def fill_time_metadata(self, time, metadata):
        step = metadata.pop(self.step_coordinate.variable.name)
        assert isinstance(step, datetime.timedelta)
        base = time - step

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = base.strftime("%Y%m%d")
        metadata["time"] = base.strftime("%H%M")
        metadata["step"] = int(hours)


class ForecastFromValidTimeAndBaseTime(Time):
    def __init__(self, forecast_reference_time):
        self.forecast_reference_time = forecast_reference_time
        assert isinstance(self.forecast_reference_time, datetime.datetime)

    def fill_time_metadata(self, time, metadata):

        step = time - self.forecast_reference_time

        hours = step.total_seconds() / 3600
        assert int(hours) == hours

        metadata["date"] = self.forecast_reference_time.strftime("%Y%m%d")
        metadata["time"] = self.forecast_reference_time.strftime("%H%M")
        metadata["step"] = int(hours)
