# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import sys
from typing import Any

from anemoi.datasets.usage.store import dataset_lookup

from . import Command

LOG = logging.getLogger(__name__)


class Which(Command):
    """Prints the path to a dataset given its name."""

    timestamp = False

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        command_parser.add_argument("path", metavar="DATASET")

    def run(self, args: Any) -> None:

        path = dataset_lookup(args.path, fail=False)
        if path is None:
            print(f"Dataset '{args.path}' not found.", file=os.sys.stderr)
            sys.exit(1)

        print(path)


command = Which
