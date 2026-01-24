# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..source import Source
from . import source_registry
from pydantic import BaseModel, Field

LOG = logging.getLogger(__name__)


class _Schema(BaseModel):
    path: str


@source_registry.register("csv")
class CSVSource(Source):
    """A source that reads data from a CSV file."""

    schema = _Schema

    emoji = "📄"  # For tracing

    def __init__(
        self,
        context: any,
        path: str,
        columns: list = None,
        flavour: dict = None,
        *args,
        **kwargs,
    ):
        """Initialise the CSVSource.

        Parameters
        ----------
        context : Any
            The context for the data source.
        filepath : str
            The path to the CSV file.
        columns : list, optional
            The list of columns to read from the CSV file.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, *args, **kwargs)

        self.path = path
        self.columns = columns

        self.flavour = {
            "latitude": "latitude",
            "longitude": "longitude",
            "date": "date",
        }

        if flavour is not None:
            self.flavour.update(flavour)

        if not isinstance(self.flavour["date"], (list, tuple)):
            self.flavour["date"] = self.flavour["date"].split(",")

    def execute(self, dates):
        import pandas as pd

        to_drop = []

        if self.columns is None:
            frame = pd.read_csv(self.path)
        else:
            frame = pd.read_csv(self.path, usecols=self.columns)

        match len(self.flavour["date"]):
            case 1:
                self.make_date_column_1(frame, self.flavour["date"][0], to_drop)
            case 2:
                self.make_date_column_2(frame, self.flavour["date"][0], self.flavour["date"][1], to_drop)
            case _:
                raise ValueError(f"Invalid number of date columns specified in flavour. ({len(self.flavour['date'])=})")

        self.make_lat_lon_columns(frame, "latitude", to_drop)
        self.make_lat_lon_columns(frame, "longitude", to_drop)

        if to_drop:
            frame.drop(columns=to_drop, inplace=True)

        mask = (frame["date"] >= dates.start_range) & (frame["date"] <= dates.end_range)

        frame = frame.loc[mask]

        return frame

    def make_lat_lon_columns(self, frame, name, to_drop):
        frame[name] = frame[self.flavour[name]].astype(float)
        if self.flavour[name] != name:
            to_drop.append(self.flavour[name])

    def make_date_column_1(self, frame, date_col, to_drop):
        import pandas as pd

        if "date" in frame.columns and date_col != "date":
            LOG.warning(f"Column 'date' already exists in data frame. Overwriting with '{date_col}'.")
            to_drop.append(date_col)

        frame["date"] = pd.to_datetime(frame[date_col])

    def make_date_column_2(self, frame, date_col, time_col, to_drop):
        import pandas as pd

        if "date" in frame.columns:
            LOG.warning(f"Column 'date' already exists in data frame. Overwriting with '{date_col}' and '{time_col}'.")

        # TODO: Read from format from flavour
        frame[time_col] = frame[time_col].astype(str).str.zfill(6)

        frame["date"] = pd.to_datetime(frame[date_col].astype(str) + " " + frame[time_col].astype(str))

        to_drop.append(date_col)
        to_drop.append(time_col)
