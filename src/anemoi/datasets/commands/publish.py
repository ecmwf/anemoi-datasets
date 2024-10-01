import logging

from . import Command

LOG = logging.getLogger(__name__)


class Publish(Command):
    """Publish a dataset."""

    # This is a command that is used to publish a dataset.
    # it is a class, inheriting from Command.

    internal = True
    timestamp = True

    def add_arguments(self, parser):
        parser.add_argument("path", help="Path of the dataset to publish.")

    def run(self, args):
        try:
            from anemoi.registry import publish_dataset
        except ImportError:
            LOG.error("anemoi-registry is not installed. Please install it to use this command.")
            return

        publish_dataset(args.path)


command = Publish
