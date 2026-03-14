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

from anemoi.datasets.usage.analytics import analytics_options
from anemoi.datasets.usage.analytics import analytics_rest_options
from anemoi.datasets.usage.store import open_zarr_store

from . import Command

LOG = logging.getLogger(__name__)


class Analytics(Command):
    """Allow to opt-in or opt-out of anemoi.datasets analytics."""

    timestamp = False

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        group = command_parser.add_mutually_exclusive_group()
        group.add_argument("--enable", action="store_true", default=False, help="Opt-in to analytics.")
        group.add_argument("--disable", action="store_true", default=False, help="Opt-out of analytics.")

        command_parser.add_argument(
            "--anemoi-user",
            help="Set a name/email/userid that can be used associate the analytics to a particular person accross several sites. To clear, just set it to 'unknown'",
        )

        command_parser.add_argument(
            "--test", metavar="DATASET", help="Show the message that will be send to the analytics server"
        )

    def run(self, args: Any) -> None:

        options = analytics_options()

        match (args.enable, args.disable):
            case (True, _):
                options["enabled"] = True
                analytics_options(options)

            case (_, True):
                options["enabled"] = False
                analytics_options(options)

        match options.get("enabled"):
            case None:
                print("Analytics not yet configured (use --enable or --disable).")
            case True:
                print("Analytics enabled. Use option --disable to opt-out.")
            case False:
                print("Analytics disabled. Use option --disable to opt-in.")

        if args.anemoi_user:
            options["anemoi_user"] = args.anemoi_user
            analytics_options(options)

        user = options.get("anemoi_user", "unknown")
        print(f"Analytics anemoi_user set to '{user}'. Change with --anemoi-user.")

        if args.test:
            print(f"Analytics that would be sent to when opening {args.test}")
            print(f"URL: {analytics_rest_options()['url']}")
            open_zarr_store(args.test, print_analytics_only=True)


command = Analytics
