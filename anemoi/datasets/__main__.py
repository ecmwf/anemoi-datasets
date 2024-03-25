#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import argparse
import logging
import sys
import traceback

from . import __version__
from .commands import COMMANDS

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="show the version and exit",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Debug mode",
    )

    subparsers = parser.add_subparsers(help="commands:", dest="command")
    for name, command in COMMANDS.items():
        command_parser = subparsers.add_parser(name, help=command.__doc__)
        command.add_arguments(command_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        return

    if args.command is None:
        parser.print_help()
        return

    cmd = COMMANDS[args.command]

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    try:
        cmd.run(args)
    except ValueError as e:
        traceback.print_exc()
        LOG.error("\n💣 %s", str(e).lstrip())
        LOG.error("💣 Exiting")
        sys.exit(1)


if __name__ == "__main__":
    main()
