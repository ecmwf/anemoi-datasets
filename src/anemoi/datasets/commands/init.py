import logging
import time

from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets.commands.create import task

from . import Command

LOG = logging.getLogger(__name__)


class Init(Command):
    """Create a dataset, step by step."""

    internal = True
    timestamp = True

    def add_arguments(self, init):

        init.add_argument("config", help="Configuration yaml file defining the recipe to create the dataset.")
        init.add_argument("path", help="Path to store the created data.")

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

        init.add_argument("--trace", action="store_true")

    def run(self, args):
        options = vars(args)
        options.pop("command")
        now = time.time()

        if "version" in options:
            options.pop("version")

        if "debug" in options:
            options.pop("debug")

        task("init", options)

        LOG.info(f"Create step '{self.__class__.__name__.lower()}' completed in {seconds_to_human(time.time()-now)}")


command = Init
