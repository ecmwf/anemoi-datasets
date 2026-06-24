# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import earthkit.data as ekd

from anemoi.datasets.create.types import DateList

from ..source import Source
from . import source_registry
from .xarray_support import XarrayFieldList
from .xarray_support import load_many
from .xarray_support import load_many_forecast
from .xarray_support import load_many_forecast_intervals
from .xarray_support import load_one

__all__ = [
    "load_many",
    "load_many_forecast",
    "load_many_forecast_intervals",
    "load_one",
    "XarrayFieldList",
]


class XarraySourceBase(Source):
    """An Xarray base data source, intended to be subclassed.

    Handles both the analysis layout (``execute_valid_dates``) and the
    trajectory layout (``execute_forecast_dates``): in the latter the file is
    selected from the forecast basetime and the step is recovered per field.
    """

    emoji = "✖️"  # For tracing

    options: dict[str, Any] | None = None
    flavour: dict[str, Any] | None = None
    patch: dict[str, Any] | None = None

    path_or_url: str | None = None

    def __init__(self, context: Any, path: str = None, url: str = None, *args: Any, **kwargs: Any):
        """Initialise the source.

        Parameters
        ----------
        context : Any
            The context for the data source.
        path : str, optional
            Path (or ``{date}`` pattern) to the dataset; mutually exclusive
            with ``url``.
        url : str, optional
            URL (or ``{date}`` pattern) to the dataset; mutually exclusive
            with ``path``.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments; field-selection criteria (e.g.
            ``param``) are forwarded to the loader.
        """
        if path is not None and url is not None:
            raise ValueError("Cannot specify both path and url")

        if path is not None:
            self.path_or_url = path
        elif url is not None:
            self.path_or_url = url

        # ``options``, ``flavour`` and ``patch`` are xarray-specific knobs that
        # some recipes set directly. Pull them out of kwargs (falling back to
        # any subclass class-level default) so they do not collide with the
        # explicit keyword arguments passed to the loaders; the rest of kwargs
        # is forwarded as field-selection criteria.
        self.options = kwargs.pop("options", self.options)
        self.flavour = kwargs.pop("flavour", self.flavour)
        self.patch = kwargs.pop("patch", self.patch)

        super().__init__(context, *args, **kwargs)

        self.args = args
        self.kwargs = kwargs

    def execute_valid_dates(self, dates: DateList) -> ekd.FieldList:
        """Execute the data loading process for the given dates.

        Parameters
        ----------
        dates : DateList
            List of dates for which data needs to be loaded.

        Returns
        -------
        ekd.FieldList
            The loaded data fields.
        """
        return load_many(
            self.emoji,
            self.context,
            dates,
            self.path_or_url,
            options=self.options,
            flavour=self.flavour,
            patch=self.patch,
            **self.kwargs,
        )

    def execute_forecast_dates(self, dates: Any) -> ekd.FieldList:
        """Load forecast fields for the trajectory layout.

        Each ``(valid_time, basetime)`` pair is resolved against a dataset
        located by substituting the basetime into the ``path``/``url`` pattern;
        the requested validity times are then selected within each dataset and
        tagged with forecast ``date``/``time``/``step`` metadata.

        Parameters
        ----------
        dates : ForecastDates
            The ``(valid_time, basetime)`` pairs for this group.

        Returns
        -------
        ekd.FieldList
            The loaded forecast fields.
        """
        return load_many_forecast(
            self.emoji,
            self.context,
            dates,
            self.path_or_url,
            options=self.options,
            flavour=self.flavour,
            patch=self.patch,
            **self.kwargs,
        )

    def execute_forecast_intervals(self, argument: Any) -> ekd.FieldList:
        """Load source increment fields for forecast accumulation windows.

        Used when ``AccumulateSource`` wraps this source in the trajectory
        layout: each covering interval is served by the field at its validity
        time, tagged with the interval metadata the accumulator needs.

        Parameters
        ----------
        argument : ForecastIntervals
            The accumulation intervals (``argument.intervals``).

        Returns
        -------
        ekd.FieldList
            The source increment fields.
        """
        return load_many_forecast_intervals(
            self.emoji,
            self.context,
            argument.intervals,
            self.path_or_url,
            options=self.options,
            flavour=self.flavour,
            patch=self.patch,
            **self.kwargs,
        )


@source_registry.register("xarray")
class XarraySource(XarraySourceBase):
    """Read fields from any xarray-openable dataset (one path/url or a ``{date}`` pattern)."""

    emoji = "🌐"
