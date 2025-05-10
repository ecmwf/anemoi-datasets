# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from anemoi.datasets.verify import verify_dataset

from . import Command

LOG = logging.getLogger(__name__)


class Verify(Command):
    """Command to inspect a zarr dataset."""

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """

        command_parser.add_argument("--callable", metavar="DATASET", default="anemoi.datasets.open_dataset")
        command_parser.add_argument("--costly-checks", action="store_true", help="Run costly checks")
        command_parser.add_argument("--detailed", action="store_true", help="Give detailed report")
        command_parser.add_argument("path", metavar="DATASET")

    def run(self, args: Any) -> None:
        """Run the command.

        Parameters
        ----------
        args : Any
            The command arguments.
        """

        package = args.callable.split(".")
        module = __import__(".".join(package[:-1]), fromlist=[package[-1]])
        callable_func = getattr(module, package[-1])

        if args.path == "default":
            args.path = "aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8"

        dataset = callable_func(args.path)
        verify_dataset(dataset, costly_checks=args.costly_checks, detailed=args.detailed)


command = Verify
