# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod

import earthkit.data as ekd

from anemoi.datasets.create.types import DateList


def reject_trajectory_dates(dates, source_name):
    """Raise ``NotImplementedError`` if *dates* are trajectory dates.

    Call this at the top of ``execute()`` in sources that do not support
    the trajectory layout.
    """
    from anemoi.datasets.create.arguments import ForecastDates
    from anemoi.datasets.create.arguments import ForecastIntervals

    if isinstance(dates, (ForecastDates, ForecastIntervals)):
        raise NotImplementedError(
            f"'{source_name}' does not support the trajectory layout. "
            f"Received {type(dates).__name__} but this source only handles plain dates (ValidDates)."
        )


class Source(ABC):
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

    @abstractmethod
    def execute(self, dates: DateList) -> ekd.FieldList:
        """Execute the filter.

        Parameters
        ----------
        dates : DateList
            The input dates.

        Returns
        -------
        ekd.FieldList
            The output data.
        """

        pass


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
