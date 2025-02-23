# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import time
from typing import Any

from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets.commands.create import task

from . import Command

LOG = logging.getLogger(__name__)


class FinaliseAdditions(Command):
    """Create a dataset, step by step."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add command line arguments to the parser.

        Parameters
        ----------
        command_parser : Any
            The argument parser instance to which arguments will be added.
        """
        command_parser.add_argument(
            "--delta",
            help="Compute statistics tendencies on a given time delta, if possible. Must be a multiple of the frequency.",
            nargs="+",
        )

        command_parser.add_argument("path", help="Path to store the created data.")
        command_parser.add_argument("--trace", action="store_true")

    def run(self, args: Any) -> None:
        """Execute the command with the given arguments.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """
        options = vars(args)
        options.pop("command")
        step = "finalise-additions"
        now = time.time()

        if "version" in options:
            options.pop("version")

        if "debug" in options:
            options.pop("debug")
            task(step, options)

        LOG.info(f"Create step '{step}' completed in {seconds_to_human(time.time()-now)}")


command = FinaliseAdditions
