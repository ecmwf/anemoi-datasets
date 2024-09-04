import logging

import yaml

from anemoi.datasets.schema import validate

from . import Command

LOG = logging.getLogger(__name__)


class Validate(Command):
    """Validate a dataset creation recipe."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):

        command_parser.add_argument("config", help="Configuration yaml file defining the recipe to create the dataset.")

    def run(self, args):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        if not validate(config):
            exit(1)


command = Validate
