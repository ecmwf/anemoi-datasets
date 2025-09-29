# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from ..source import ObservationsSource
from . import source_registry


@source_registry.register("csv")
class CSVSource(ObservationsSource):
    """A source that reads data from a CSV file."""

    emoji = "📄"  # For tracing

    def __init__(self, context: any, path: str, *args: tuple, **kwargs: dict):
        """Initialise the CSVSource.

        Parameters
        ----------
        context : Any
            The context for the data source.
        filepath : str
            The path to the CSV file.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, *args, **kwargs)
        self.path = path

    def execute(self, dates):
        import pandas as pd

        frame = pd.read_csv(self.path)
        print(frame)
