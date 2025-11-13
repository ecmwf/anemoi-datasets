# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import codc
import pandas

from anemoi.datasets.create.gridded.typing import DateList

from ..source import Source
from . import source_registry


@source_registry.register("odb")
class OdbSource(Source):
    """ODB data source."""

    emoji = "🔭"

    def __init__(
        self,
        context,
        path: str,
        select: str,
        where: str,
        flavour: str,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialise the ODB input.

        Parameters
        ----------
        context : dict
            The context.
        path : str
            The path to the ODB file.
        select : str
            The select clause.
        where : str
            The where clause.
        flavour : str
            The naming of the latitude, longitude, date and time columns.
        kwargs : dict, optional
            Additional keyword arguments.

        """
        super().__init__(context)

        self.path = path
        self.select = select
        self.where = where
        self.flavour = flavour

    def execute(self, dates: DateList) -> pandas.dataframe.DataFrame:
        """Execute the ODB source.

        Parameters
        ----------
        dates : DateList
            The input dates.

        Returns
        -------
        pandas.dataframe.DataFrame
            The output dataframe.
        """
        # Code to load ODB files goes here.
        df = codc.read_odb(self.path, select=self.select, where=self.where, dates=dates)
        return df
