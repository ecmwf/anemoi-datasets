import datetime
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import tqdm
from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets.create.trace import enable_trace

from . import Command

LOG = logging.getLogger(__name__)


from anemoi.datasets.commands.create import task


class CreateStep(Command):
    """Create a dataset, step by step."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        subparsers = command_parser.add_subparsers(dest="step")
        init = subparsers.add_parser("init")
        load = subparsers.add_parser("load")
        statistics = subparsers.add_parser("statistics")
        finalise = subparsers.add_parser("finalise")
        size = subparsers.add_parser("size")
        patch = subparsers.add_parser("patch")
        init_additions = subparsers.add_parser("init-additions")
        run_additions = subparsers.add_parser("run-additions")
        finalise_additions = subparsers.add_parser("finalise-additions")
        cleanup = subparsers.add_parser("cleanup")
        verify = subparsers.add_parser("verify")
        all = [
            init,
            load,
            statistics,
            finalise,
            size,
            patch,
            init_additions,
            run_additions,
            finalise_additions,
            cleanup,
            verify,
        ]

        init.add_argument("config", help="Configuration yaml file defining the recipe to create the dataset.")
        init.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files. This will delete the target dataset if it already exists.",
        )
        init.add_argument(
            "--test",
            action="store_true",
            help="Build a small dataset, using only the first dates. And, when possible, using low resolution and less ensemble members.",
        )
        init.add_argument(
            "--check-name",
            dest="check_name",
            action="store_true",
            help="Check if the dataset name is valid before creating it.",
        )
        init.add_argument(
            "--no-check-name",
            dest="check_name",
            action="store_false",
            help="Do not check if the dataset name is valid before creating it.",
        )
        init.set_defaults(check_name=False)

        load.add_argument("--parts", nargs="+", help="Only load the specified parts of the dataset.")
        run_additions.add_argument("--parts", nargs="+", help="Only run the specified parts of the dataset.")

        for subparser in [init_additions, run_additions, finalise_additions]:
            subparser.add_argument(
                "--delta",
                nargs="+",
                type=int,
                help="Only run the specified deltas of the dataset.",
                default=[1, 3, 6, 12, 24],
            )

        for subparser in all:
            subparser.add_argument("path", help="Path to store the created data.")
            subparser.add_argument("--trace", action="store_true")

    def run(self, args):
        options = vars(args)
        options.pop("command")
        step = options.pop("step")
        now = time.time()
        task(step, options)
        LOG.info(f"Create step '{step}' completed in {seconds_to_human(time.time()-now)}")


command = CreateStep
