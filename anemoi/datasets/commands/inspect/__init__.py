# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os

from .. import Command

# from .checkpoint import InspectCheckpoint
from .zarr import InspectZarr


class Inspect(Command, InspectZarr):
    # class Inspect(Command, InspectCheckpoint, InspectZarr):
    """Inspect a checkpoint or zarr file."""

    def add_arguments(self, command_parser):
        # g = command_parser.add_mutually_exclusive_group()
        # g.add_argument("--inspect", action="store_true", help="Inspect weights")
        command_parser.add_argument("path", metavar="PATH", nargs="+")
        command_parser.add_argument("--detailed", action="store_true")
        # command_parser.add_argument("--probe", action="store_true")
        command_parser.add_argument("--progress", action="store_true")
        command_parser.add_argument("--statistics", action="store_true")
        command_parser.add_argument("--size", action="store_true", help="Print size")

    def run(self, args):
        dic = vars(args)
        for path in dic.pop("path"):
            if os.path.isdir(path) or path.endswith(".zarr.zip") or path.endswith(".zarr"):
                self.inspect_zarr(path=path, **dic)
            else:
                raise ValueError(f"Unknown file type: {path}")
                # self.inspect_checkpoint(path=path, **dic)


command = Inspect
