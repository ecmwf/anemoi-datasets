# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import TYPE_CHECKING

from anemoi.transform import FieldList

from anemoi.datasets.create.types import DateList

if TYPE_CHECKING:
    from anemoi.datasets.create.arguments import ForecastDates
    from anemoi.datasets.create.arguments import ForecastIntervals
    from anemoi.datasets.create.arguments import Intervals
    from anemoi.datasets.create.arguments import ValidDates


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

    def execute(self, argument) -> FieldList:
        """Dispatch to the appropriate ``execute_*`` method based on argument type.

        Plain ``list[datetime]`` and ``GroupOfDates`` are wrapped in
        ``ValidDates`` for backward compatibility.

        Parameters
        ----------
        argument : ValidDates | ForecastDates | Intervals | ForecastIntervals
            The typed argument from the pipeline.

        Returns
        -------
        FieldList
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
            f"{type(self).__name__}.execute() received unsupported argument type " f"'{type(argument).__name__}'."
        )

    def execute_valid_dates(self, argument: "ValidDates") -> FieldList:
        """Handle instant analysis / reanalysis requests.

        Override in subclasses to process ``ValidDates`` arguments.

        Parameters
        ----------
        argument : ValidDates
            The validity-time argument from the pipeline.

        Returns
        -------
        FieldList
            The output data.
        """
        raise NotImplementedError(f"'{type(self).__name__}' does not implement execute_valid_dates.")

    def execute_forecast_dates(self, argument: "ForecastDates") -> FieldList:
        """Handle forecast (basetime, valid_time) requests — trajectories / step products.

        Override in subclasses to process ``ForecastDates`` arguments.
        By default raises ``NotImplementedError`` (source does not support trajectories).

        Parameters
        ----------
        argument : ForecastDates
            The forecast-date argument from the pipeline.

        Returns
        -------
        FieldList
            The output data.
        """
        raise NotImplementedError(
            f"'{type(self).__name__}' does not support the trajectory layout " f"(received {type(argument).__name__})."
        )

    def execute_intervals(self, argument: "Intervals") -> FieldList:
        """Handle archived-interval requests from the windowed sources.

        ``accumulate`` (and ``maximum`` / ``minimum`` / ``average`` over
        interval-valued source data) ask for the archived fields covering each
        window, which only a source that can address a field by the interval it
        spans can serve. There is no fallback: ``execute_valid_dates`` would
        receive only the output dates and miss most of every window.

        Parameters
        ----------
        argument : Intervals
            The interval argument from the pipeline.

        Returns
        -------
        FieldList
            The output data.
        """
        raise NotImplementedError(self._cannot_serve_intervals())

    def execute_forecast_intervals(self, argument: "ForecastIntervals") -> FieldList:
        """Handle archived-interval requests for the run from the layout (``base_dates: from-layout``).

        As :meth:`execute_intervals`, with no fallback either.

        Parameters
        ----------
        argument : ForecastIntervals
            The forecast-interval argument from the pipeline.

        Returns
        -------
        FieldList
            The output data.
        """
        raise NotImplementedError(self._cannot_serve_intervals(from_layout=True))

    def _cannot_serve_intervals(self, from_layout: bool = False) -> str:
        layout = " for the run from the layout (base_dates: from-layout)" if from_layout else ""
        supported = "'mars', 'grib' and 'grib-index'" if from_layout else "'mars', 'fdb', 'grib' and 'grib-index'"
        return (
            f"'{type(self).__name__}' cannot serve archived intervals{layout}: a window over fields that "
            f"each span an interval needs a source that can address a field by that interval, and only "
            f"{supported} can. Use one of those as the 'source:', or, for instantaneous fields, "
            "'from: {frequency: ...}' under 'maximum:' / 'minimum:' / 'average:'."
        )


class FieldSource(Source):
    """A source that returns a predefined FieldList."""

    def __init__(self, context: any, data: FieldList, *args: tuple, **kwargs: dict):
        """Initialise the FieldSource.

        Parameters
        ----------
        context : Any
            The context for the data source.
        data : FieldList
            The predefined data to return.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, *args, **kwargs)
        self.data = data

    def execute(self, dates: DateList) -> FieldList:
        """Return the predefined FieldList.

        Parameters
        ----------
        dates : DateList
            The input dates (not used in this implementation).

        Returns
        -------
        FieldList
            The predefined data.
        """
        self.context.trace(self.emoji, f"FieldSource returning {len(self.data)} fields")
        return self.data


class ObservationsSource(Source):
    """A source that retrieves observational data."""

    pass
