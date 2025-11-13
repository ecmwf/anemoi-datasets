# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from ..source import Source
from . import source_registry


@source_registry.register("csv")
class CSVSource(Source):
    """A source that reads data from a CSV file."""

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
            "time": "time",
        }

        if flavour is not None:
            self.flavour.update(flavour)

    def execute(self, dates):
        import pandas as pd

        if self.columns is None:
            frame = pd.read_csv(self.path)
        else:
            frame = pd.read_csv(self.path, usecols=self.columns)

        print(sorted(frame.columns))

        mask = (frame[self.flavour["time"]] >= dates.start_date) & (frame[self.flavour["time"]] <= dates.end_date)

        frame = frame.loc[mask]

        return frame
