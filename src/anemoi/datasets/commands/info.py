import json
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

        for a in z.arrays():
            print(a)

        # print(list(z.arrays()))

        # print(dir(z))
        print(json.dumps(dict(z.attrs), indent=2))


command = Info
