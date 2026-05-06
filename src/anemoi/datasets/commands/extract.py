# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from . import Command

LOG = logging.getLogger(__name__)


class Extract(Command):
    """Extract constants and/or climatologies from a dataset."""

    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        command_parser.add_argument(
            "--constant",
            action="append",
            default=[],
            help="Name of a constant variable to extract. Can be repeated.",
        )
        command_parser.add_argument(
            "--climatology",
            action="append",
            default=[],
            help="Name of a climatology variable to extract. Can be repeated.",
        )
        command_parser.add_argument(
            "--format",
            choices=["netcdf"],
            default="netcdf",
            help="Output format (default: netcdf).",
        )
        command_parser.add_argument("dataset", metavar="DATASET", help="Path or name of the dataset.")
        command_parser.add_argument("output", metavar="OUTPUT", help="Output file path.")

    def run(self, args: Any) -> None:

        from anemoi.datasets import open_dataset

        ds = open_dataset(args.dataset)

        constants = args.constant
        climatologies = args.climatology

        if not constants and not climatologies:
            raise ValueError("At least one --constant or --climatology must be specified.")

        constant_data = {}
        climatology_data = {}

        if constants:
            for name in constants:
                constant_data[name] = ds[ds.to_index(0, name)]

        if climatologies:
            monthly_dates = self._select_monthly_dates(ds)
            for name in climatologies:
                values = []
                for date in monthly_dates:
                    values.append(ds[ds.to_index(date, name)])
                climatology_data[name] = values

        if args.format == "netcdf":
            self._write_netcdf(args.output, constant_data, climatology_data, ds)

    def _select_monthly_dates(self, ds: Any) -> list:
        """Select one date per month of the year from the dataset.

        Parameters
        ----------
        ds : Any
            The dataset.

        Returns
        -------
        list
            A list of 12 dates, one for each month.
        """
        import numpy as np

        dates = ds.dates
        selected = {}
        for date in dates:
            dt = date.astype("datetime64[ms]").astype("datetime64[M]")
            month = int(str(dt).split("-")[1])
            if month not in selected:
                selected[month] = np.datetime64(date)
        if len(selected) < 12:
            raise ValueError(
                f"Dataset does not cover all 12 months (found {len(selected)}). " "Cannot compute climatologies."
            )
        return [selected[m] for m in range(1, 13)]

    def _write_netcdf(self, output: str, constant_data: dict, climatology_data: dict, ds: Any) -> None:
        import datetime

        import numpy as np

        try:
            from netCDF4 import Dataset as NetCDFDataset
        except ImportError:
            raise ImportError("netCDF4 is required for netcdf output. Install it with: pip install netCDF4")

        num_points = len(ds.latitudes)
        variables_metadata = ds.variables_metadata

        with NetCDFDataset(output, "w", format="NETCDF4") as nc:
            nc.createDimension("values", num_points)

            lat_var = nc.createVariable("latitude", np.float64, ("values",))
            lat_var[:] = ds.latitudes
            lat_var.units = "degrees_north"

            lon_var = nc.createVariable("longitude", np.float64, ("values",))
            lon_var[:] = ds.longitudes
            lon_var.units = "degrees_east"

            # Write constants
            for name, data in constant_data.items():
                var = nc.createVariable(name, np.float32, ("values",))
                var[:] = data
                units = variables_metadata.get(name, {}).get("units")
                if units:
                    var.units = units

            # Write climatologies with CF conventions
            if climatology_data:
                nc.createDimension("time", 12)
                nc.createDimension("nv", 2)

                # Time variable: mid-month days since 2000-01-01
                ref_date = datetime.datetime(2000, 1, 1)
                time_values = []
                clim_bounds = []
                for month in range(1, 13):
                    # Mid-month as representative time
                    mid = datetime.datetime(2000, month, 15)
                    time_values.append((mid - ref_date).days)
                    # Bounds: first and last day of the month
                    start = datetime.datetime(2000, month, 1)
                    if month == 12:
                        end = datetime.datetime(2001, 1, 1)
                    else:
                        end = datetime.datetime(2000, month + 1, 1)
                    clim_bounds.append([(start - ref_date).days, (end - ref_date).days])

                time_var = nc.createVariable("time", np.float64, ("time",))
                time_var[:] = time_values
                time_var.units = "days since 2000-01-01"
                time_var.climatology = "climatology_bounds"
                time_var.calendar = "gregorian"

                clim_bounds_var = nc.createVariable("climatology_bounds", np.float64, ("time", "nv"))
                clim_bounds_var[:] = clim_bounds

                for name, monthly_values in climatology_data.items():
                    var = nc.createVariable(name, np.float32, ("time", "values"))
                    for i, data in enumerate(monthly_values):
                        var[i, :] = data
                    var.cell_methods = "time: mean within years time: mean over years"
                    units = variables_metadata.get(name, {}).get("units")
                    if units:
                        var.units = units


command = Extract
