# (C) Copyright 2024-2026 Anemoi contributors.
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


class Finalise(Command):
    """Create a dataset, step by step."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser to which arguments will be added.
        """
        command_parser.add_argument("path", help="Path to store the created data.")
        command_parser.add_argument("--trace", action="store_true")

        # Run a single finalise stage so the zarr can be populated incrementally from
        # several processes. With no flag, all stages run in one process (backward
        # compatible). `--load` selects fragments with `--parts` (1-based, like `load`).
        stage = command_parser.add_mutually_exclusive_group()
        stage.add_argument(
            "--prepare",
            dest="finalise_stage",
            action="store_const",
            const="prepare",
            help="Deduplicate, compute the shape, create the zarr array and write the manifest.",
        )
        stage.add_argument(
            "--rows-per-chunk",
            dest="finalise_stage",
            action="store_const",
            const="rows_per_chunk",
            help="Compute and print the optimal rows-per-chunk for each iteration window (report only).",
        )
        stage.add_argument(
            "--load",
            dest="finalise_stage",
            action="store_const",
            const="load",
            help="Write the fragments of the given --parts into the zarr array.",
        )
        stage.add_argument(
            "--tidy",
            dest="finalise_stage",
            action="store_const",
            const="tidy",
            help="Merge statistics and date ranges, build the index and clean up.",
        )
        command_parser.add_argument(
            "--parts",
            nargs="+",
            help="Only load the specified parts of the dataset (1-based, e.g. '2/5'). Used with --load.",
        )
        command_parser.add_argument(
            "--print",
            dest="rows_per_chunk_print",
            action="store_true",
            help="With --rows-per-chunk: just print the optimal rows-per-chunk for every window, "
            "without changing the dataset.",
        )

    def run(self, args: Any) -> None:
        """Execute the finalise command.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """
        options = vars(args)
        options.pop("command")
        now = time.time()
        step = "finalise"

        if "version" in options:
            options.pop("version")

        if "debug" in options:
            options.pop("debug")

        task(step, options)

        LOG.info(f"Create step '{step}' completed in {seconds_to_human(time.time()-now)}")


command = Finalise
