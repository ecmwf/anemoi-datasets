import logging

from anemoi.datasets.data.stores import open_zarr
from anemoi.datasets.data.stores import zarr_lookup

from . import Command

LOG = logging.getLogger(__name__)


class Info(Command):
    """Dump information about a dataset."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument("path", help="Path to a dataset.")

    def run(self, args):

        z = open_zarr(zarr_lookup(args.path))
        print(dict(z.attrs))

        for k, v in z:
            print(k, v)


command = Info
