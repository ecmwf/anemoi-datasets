from anemoi.datasets.create import Creator

from . import Command


class Create(Command):
    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
        command_parser.add_argument("config", help="Configuration file")
        command_parser.add_argument("path", help="Path to store the created data")

    def run(self, args):
        kwargs = vars(args)

        c = Creator(**kwargs)
        c.create()


command = Create
