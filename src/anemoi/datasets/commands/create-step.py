import logging
import time

from anemoi.utils.humanize import seconds_to_human


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
        size = subparsers.add_parser("size")
        finalise = subparsers.add_parser("finalise")  # finalize = statistics + size (for now)

        init_additions = subparsers.add_parser("init-additions")
        run_additions = subparsers.add_parser("run-additions")
        finalise_additions = subparsers.add_parser("finalise-additions")
        additions = subparsers.add_parser(
            "additions"
        )  # additions = init_additions + run_additions + finalise_additions

        patch = subparsers.add_parser("patch")
        cleanup = subparsers.add_parser("cleanup")
        verify = subparsers.add_parser("verify")

        ALL = [
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
            additions,
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

        for s in [load, run_additions]:
            s.add_argument("--parts", nargs="+", help="Only load the specified parts of the dataset.")

        for s in [init_additions, run_additions, finalise_additions, additions]:
            s.add_argument(
                "--delta",
                help="Compute statistics tendencies on a given time delta, if possible. Must be a multiple of the frequency.",
            )

        for s in ALL:
            s.add_argument("path", help="Path to store the created data.")
            s.add_argument("--trace", action="store_true")

    def run(self, args):
        options = vars(args)
        options.pop("command")
        step = options.pop("step")
        now = time.time()

        if "version" in options:
            options.pop("version")

        if "debug" in options:
            options.pop("debug")
            task(step, options)

        LOG.info(f"Create step '{step}' completed in {seconds_to_human(time.time()-now)}")


command = CreateStep
