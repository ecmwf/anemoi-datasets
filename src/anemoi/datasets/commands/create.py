import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from . import Command

LOG = logging.getLogger(__name__)


def task(what, options, *args, **kwargs):
    """
    Make sure `import Creator` is done in the sub-processes, and not in the main one.
    """

    now = datetime.datetime.now()
    LOG.info(f"Task {what}({args},{kwargs}) starting")

    from anemoi.datasets.create import Creator

    c = Creator(**options)
    getattr(c, what)(*args, **kwargs)

    LOG.info(f"Task {what}({args},{kwargs}) completed ({datetime.datetime.now()-now})")


class Create(Command):
    """Create a dataset."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files. This will delete the target dataset if it already exists.",
        )
        command_parser.add_argument(
            "--test",
            action="store_true",
            help="Build a small dataset, using only the first dates. And, when possible, using low resolution and less ensemble members.",
        )
        command_parser.add_argument("config", help="Configuration yaml file defining the recipe to create the dataset.")
        command_parser.add_argument("path", help="Path to store the created data.")
        command_parser.add_argument("--parallel", help="Use `n` parallel workers.", type=int, default=0)
        command_parser.add_argument("--use-threads", action="store_true")

    def run(self, args):
        if args.parallel:
            self.parallel_create(args)
        else:
            self.serial_create(args)

    def serial_create(self, args):
        from anemoi.datasets.create import Creator

        options = vars(args)
        c = Creator(**options)
        c.create()

    def parallel_create(self, args):
        """Some modules, like fsspec do not work well with fork()
        Other modules may not be thread safe. So we implement
        parallel loadining using multiprocessing before any
        of the modules are imported.
        """

        options = vars(args)
        parallel = options.pop("parallel")

        if args.use_threads:
            ExecutorClass = ThreadPoolExecutor
        else:
            ExecutorClass = ProcessPoolExecutor

        with ExecutorClass(max_workers=1) as executor:
            executor.submit(task, "init", options).result()

        futures = []

        with ExecutorClass(max_workers=parallel) as executor:
            for n in range(parallel):
                futures.append(executor.submit(task, "load", options, parts=f"{n+1}/{parallel}"))

            for future in as_completed(futures):
                future.result()

        with ExecutorClass(max_workers=1) as executor:
            executor.submit(task, "statistics", options).result()
            executor.submit(task, "additions", options).result()
            executor.submit(task, "cleanup", options).result()
            executor.submit(task, "verify", options).result()


command = Create
