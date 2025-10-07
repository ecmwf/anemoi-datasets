# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import importlib
import logging
from typing import Any

from anemoi.datasets.validate import validate_dataset

from . import Command

LOG = logging.getLogger(__name__)

DEFAULT_DATASET = "aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8"


class Validate(Command):
    """Command to validate an anemoi dataset."""

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

        module_path, func_name = args.callable.rsplit(".", 1)
        module = importlib.import_module(module_path)
        callable_func = getattr(module, func_name)

        if args.path == "default":
            args.path = DEFAULT_DATASET

        dataset = callable_func(args.path)
        validate_dataset(dataset, costly_checks=args.costly_checks, detailed=args.detailed)


command = Validate
