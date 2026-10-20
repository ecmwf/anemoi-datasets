# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# DRAFT: untested. Written from the earthkit-data polytope docs and the FDB source.
# See the TODOs for what still has to be investigated.

from typing import Any

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.flavour import RuleBasedFlavour
from anemoi.transform.grids import grid_registry

from anemoi.datasets.create.arguments import ValidDates

from ..source import Source
from . import source_registry
from .mars import compress_prebuilt_requests


@source_registry.register("polytope")
class PolytopeSource(Source):
    """Polytope data source (full-field GRIB retrieval)."""

    emoji = "🔺"

    def __init__(
        self,
        context,
        request: dict[str, Any],
        collection: str = "destination-earth",
        address: str | None = None,
        flavour: dict[str, Any] | None = None,
        grid_definition: str | None = None,
        **kwargs: Any,
    ):
        """Initialise the Polytope input.

        Parameters
        ----------
        context : Any
            The context.
        request : dict
            The MARS-like request, without the time keys (``date``, ``time``)
            which are filled in from the dates being built.
        collection : str, optional
            The Polytope collection, e.g. ``destination-earth`` or ``ecmwf-mars``.
        address : str, optional
            The Polytope server address. For DestinE on-demand extremes this is
            ``polytope.lumi.apps.dte.destination-earth.eu``.
        flavour : dict, optional
            The flavour configuration, see `anemoi.transform.flavour.RuleBasedFlavour`.
        grid_definition : str, optional
            The grid definition to use, see `anemoi.transform.grids.grid_registry`.
        kwargs : dict, optional
            Extra keyword arguments passed to ``earthkit.data.from_source("polytope", ...)``.
        """
        super().__init__(context)
        self.request = request.copy()
        self.request.setdefault("step", 0)
        self.collection = collection
        self.polytope_kwargs = dict(kwargs)
        if address is not None:
            self.polytope_kwargs["address"] = address
        # Return data in memory, as in the Polytope examples
        self.polytope_kwargs.setdefault("stream", False)

        self.flavour = RuleBasedFlavour(flavour) if flavour else None
        self.grid = grid_registry.from_config(grid_definition) if grid_definition is not None else None

    def execute_valid_dates(self, dates: ValidDates) -> ekd.FieldList:
        """Handle instant requests."""
        # TODO: date/time/step handling is copied from the simplest FDB case. On-demand
        # extremes runs are event-triggered, so check how base times map to valid dates.
        requests = [self.request | {"date": date.strftime("%Y%m%d"), "time": date.strftime("%H%M")} for date in dates]
        return self._execute_requests(requests)

    # TODO: execute_intervals / execute_forecast_dates (accumulations, step products),
    # following FdbSource.execute_intervals.

    def _execute_requests(self, requests: list[dict[str, Any]]) -> ekd.FieldList:
        requests = compress_prebuilt_requests(requests or [self.request])

        # TODO: check Polytope request-size limits and whether compression
        # should be capped (the MARS source chunks, this does not).
        fl = ekd.from_source("empty")
        for request in requests:
            fl += ekd.from_source("polytope", self.collection, request, **self.polytope_kwargs)

        if self.grid is not None:
            fl = new_fieldlist_from_list([new_field_from_grid(f, self.grid) for f in fl])

        if self.flavour:
            fl = self.flavour.map(fl)

        return fl
