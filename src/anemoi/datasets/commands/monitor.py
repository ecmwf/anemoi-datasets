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


class Monitor(Command):
    """Plot a dataset's build diagnostics and monthly maps into <dataset>/_monitor/."""

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser to which arguments will be added.
        """
        command_parser.add_argument("path", help="Path to the dataset.")
        command_parser.add_argument("--month", help="Month of the maps (YYYY-MM); the last one by default.")
        command_parser.add_argument(
            "--variables", nargs="+", help="Variables to map; the recipe's grid_observations `maps` by default."
        )

    def run(self, args: Any) -> None:
        """Run the command.

        Parameters
        ----------
        args : Any
            The command arguments.
        """
        from anemoi.datasets.monitor import plot_all

        for out in plot_all(args.path, args.month, args.variables):
            print(out)


command = Monitor
