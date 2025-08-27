# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from typing import Any

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.flavour import RuleBasedFlavour
from anemoi.transform.grids import grid_registry

from anemoi.datasets.create.typing import DateList

from ..source import Source
from . import source_registry


@source_registry.register("fdb")
class FdbSource(Source):
    """FDB data source."""

    emoji = "💽"

    def __init__(
        self,
        context,
        fdb_request: dict[str, Any],
        fdb_config: dict | None = None,
        fdb_userconfig: dict | None = None,
        flavour: dict[str, Any] | None = None,
        grid_definition: str | None = None,
        **kwargs: dict[str, Any],
    ):
        """Initialise the FDB input.

        Parameters
        ----------
        context : dict
            The context.
        fdb_request: dict
            The FDB request parameters.
        fdb_config : dict, optional
            The FDB config to use.
        fdb_userconfig : dict, optional
            The FDB userconfig to use.
        flavour: dict, optional
            The flavour configuration, see `anemoi.transform.flavour.RuleBasedFlavour`.
        grid_definition : str, optional
            The grid definition to use, see `anemoi.transform.grids.grid_registry`.
        kwargs : dict, optional
            Additional keyword arguments.
        """
        super().__init__(context)
        self.request = fdb_request.copy()
        self.configs = {"config": fdb_config, "userconfig": fdb_userconfig}

        self.flavour = RuleBasedFlavour(flavour) if flavour else None
        if grid_definition is not None:
            self.grid = grid_registry.from_config(grid_definition)
        else:
            self.grid = None

        if "step" not in self.request:
            self.request["step"] = 0

        self.request["param"] = _shortname_to_paramid(fdb_request["param"], kwargs.pop("param_id_map", None))

        # temporary workarounds for FDB use at MeteoSwiss (adoption is ongoing)
        # thus not documented
        self.offset_from_date = kwargs.pop("offset_from_date", None)

    def execute(self, dates: DateList) -> ekd.FieldList:
        """Execute the FDB source.

        Parameters
        ----------
        dates : DateList
            The input dates.

        Returns
        -------
        ekd.FieldList
            The output data.
        """

        requests = []
        for date in dates:
            time_request = _time_request_keys(date, self.offset_from_date)
            requests.append(self.request | time_request)

        # in some cases (e.g. repeated_dates 'constant' mode), we might have a fully
        # defined request already and an empty dates list
        requests = requests or [self.request]

        fl = ekd.from_source("empty")
        for request in requests:
            fl += ekd.from_source("fdb", request, **self.configs, read_all=True)

        if self.grid is not None:
            fl = new_fieldlist_from_list([new_field_from_grid(f, self.grid) for f in fl])

        if self.flavour:
            fl = self.flavour.map(fl)

        return fl


def _time_request_keys(dt: datetime, offset_from_date: bool | None = None) -> str:
    """Defines the time-related keys for the FDB request."""
    out = {}
    out["date"] = dt.strftime("%Y%m%d")
    if offset_from_date:
        out["time"] = "0000"
        out["step"] = int((dt - dt.replace(hour=0, minute=0)).total_seconds() // 3600)
    else:
        out["time"] = dt.strftime("%H%M")
    return out


def _shortname_to_paramid(shortname: list[str], param_id_map: dict[str, int] | None = None) -> list[int]:
    from anemoi.datasets.create.sources.mars import use_grib_paramid

    """Convert a shortname to a parameter ID."""
    if param_id_map is None:
        return use_grib_paramid(shortname)
    return [param_id_map[s] for s in shortname]
