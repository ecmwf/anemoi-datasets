# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd

from anemoi.datasets.create.types import DateList


class Source:
    """Represents a data source with a given context."""

    emoji = "📦"  # For tracing

    def __init__(self, context: any, *args: tuple, **kwargs: dict):
        """Initialise the source.

        Parameters
        ----------
        context : Any
            The context for the data source.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.context = context

    def execute(self, argument) -> ekd.FieldList:
        """Dispatch to the appropriate ``execute_*`` method based on argument type.

        Plain ``list[datetime]`` and ``GroupOfDates`` are wrapped in
        ``ValidDates`` for backward compatibility.

        Parameters
        ----------
        argument : ValidDates | ForecastDates | Intervals | ForecastIntervals
            The typed argument from the pipeline.

        Returns
        -------
        ekd.FieldList
            The output data.
        """
        from anemoi.datasets.create.arguments import ForecastDates
        from anemoi.datasets.create.arguments import ForecastIntervals
        from anemoi.datasets.create.arguments import Intervals
        from anemoi.datasets.create.arguments import ValidDates
        from anemoi.datasets.dates.groups import GroupOfDates

        if isinstance(argument, list):
            argument = ValidDates(argument)
        elif isinstance(argument, GroupOfDates):
            argument = ValidDates(argument.dates)

        # Subclasses must be checked before their parents (ForecastIntervals
        # before ForecastDates; Intervals before ValidDates).
        if isinstance(argument, ForecastIntervals):
            return self.execute_forecast_intervals(argument)
        if isinstance(argument, Intervals):
            return self.execute_intervals(argument)
        if isinstance(argument, ForecastDates):
            return self.execute_forecast_dates(argument)
        if isinstance(argument, ValidDates):
            return self.execute_valid_dates(argument)
        raise TypeError(
            f"{type(self).__name__}.execute() received unsupported argument type "
            f"'{type(argument).__name__}'."
        )

    def execute_valid_dates(self, argument: "ValidDates") -> ekd.FieldList:
        """Handle instant analysis / reanalysis requests.

        Override in subclasses to process ``ValidDates`` arguments.

        Parameters
        ----------
        argument : ValidDates
            The validity-time argument from the pipeline.

        Returns
        -------
        ekd.FieldList
            The output data.
        """
        raise NotImplementedError(f"'{type(self).__name__}' does not implement execute_valid_dates.")

    def execute_forecast_dates(self, argument: "ForecastDates") -> ekd.FieldList:
        """Handle forecast (basetime, valid_time) requests — trajectories / step products.

        Override in subclasses to process ``ForecastDates`` arguments.
        By default raises ``NotImplementedError`` (source does not support trajectories).

        Parameters
        ----------
        argument : ForecastDates
            The forecast-date argument from the pipeline.

        Returns
        -------
        ekd.FieldList
            The output data.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' does not support the trajectory layout "
            f"(received {type(argument).__name__})."
        )

    def execute_intervals(self, argument: "Intervals") -> ekd.FieldList:
        """Handle archive-resolved interval requests from ``AccumulateSource``.

        Falls back to ``execute_valid_dates`` by default. Override explicitly
        when step-range requests must be encoded (e.g. ``MarsSource``).

        Parameters
        ----------
        argument : Intervals
            The interval argument from the pipeline.

        Returns
        -------
        ekd.FieldList
            The output data.
        """
        return self.execute_valid_dates(argument)

    def execute_forecast_intervals(self, argument: "ForecastIntervals") -> ekd.FieldList:
        """Handle forecast accumulation requests.

        Falls back to ``execute_forecast_dates`` by default. Override explicitly
        when step-range requests must be encoded.

        Parameters
        ----------
        argument : ForecastIntervals
            The forecast-interval argument from the pipeline.

        Returns
        -------
        ekd.FieldList
            The output data.
        """
        return self.execute_forecast_dates(argument)


class FieldSource(Source):
    """A source that returns a predefined FieldList."""

    def __init__(self, context: any, data: ekd.FieldList, *args: tuple, **kwargs: dict):
        """Initialise the FieldSource.

        Parameters
        ----------
        context : Any
            The context for the data source.
        data : ekd.FieldList
            The predefined data to return.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, *args, **kwargs)
        self.data = data

    def execute(self, dates: DateList) -> ekd.FieldList:
        """Return the predefined FieldList.

        Parameters
        ----------
        dates : DateList
            The input dates (not used in this implementation).

        Returns
        -------
        ekd.FieldList
            The predefined data.
        """
        self.context.trace(self.emoji, f"FieldSource returning {len(self.data)} fields")
        return self.data


class ObservationsSource(Source):
    """A source that retrieves observational data."""

    pass
