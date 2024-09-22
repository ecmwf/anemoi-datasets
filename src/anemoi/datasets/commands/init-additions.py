import logging
import time

from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets.commands.create import task

from . import Command

LOG = logging.getLogger(__name__)


class InitAdditions(Command):
    """Create a dataset, step by step."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "--delta",
            help="Compute statistics tendencies on a given time delta, if possible. Must be a multiple of the frequency.",
            nargs="+",
        )

        command_parser.add_argument("path", help="Path to store the created data.")
        command_parser.add_argument("--trace", action="store_true")

    def run(self, args):
        options = vars(args)
        options.pop("command")
        step = "init-additions"
        now = time.time()

        if "version" in options:
            options.pop("version")

        if "debug" in options:
            options.pop("debug")
            task(step, options)

        LOG.info(f"Create step '{step}' completed in {seconds_to_human(time.time()-now)}")


command = InitAdditions
