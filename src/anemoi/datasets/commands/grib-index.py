# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import fnmatch
import os
from typing import Any

import tqdm

from . import Command

KEYS1 = ("class", "type", "stream", "expver", "levtype")
KEYS2 = ("shortName", "paramId", "level", "step", "number", "date", "time", "valid_datetime", "levelist")

KEYS = KEYS1 + KEYS2


class GribIndexCmd(Command):
    internal = True
    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser to which arguments are added.
        """
        command_parser.add_argument(
            "--index",
            help="Create an index file",
            required=True,
        )

        command_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Over write the index file",
        )

        command_parser.add_argument(
            "--match",
            help="Give a glob pattern to match files (default: *.grib)",
            default="*.grib",
        )

        command_parser.add_argument(
            "--flavour",
            help="GRIB flavour file (yaml or json)",
        )

        command_parser.add_argument("paths", nargs="+", help="Paths to scan")

    def run(self, args: Any) -> None:
        """Execute the scan command.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """

        def match(path: str) -> bool:
            """Check if a path matches the given pattern.

            Parameters
            ----------
            path : str
                The path to check.

            Returns
            -------
            bool
                True if the path matches, False otherwise.
            """
            return fnmatch.fnmatch(path, args.match)

        from anemoi.datasets.create.sources.grib_index import GribIndex

        # Remove namespace if present
        keys = [k.split(".")[-1] for k in KEYS]

        index = GribIndex(
            args.index,
            keys=keys,
            update=True,
            overwrite=args.overwrite,
        )

        paths = []
        for path in args.paths:
            if os.path.isfile(path):
                paths.append(path)
            else:
                for root, _, files in os.walk(path):
                    for file in files:
                        full = os.path.join(root, file)
                        paths.append(full)

        for path in tqdm.tqdm(paths, leave=False):
            if match(path):
                index.add_grib_file(path)


command = GribIndexCmd
