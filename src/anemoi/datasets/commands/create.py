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


def task(what, options, *args, **kwargs):
    """
    Make sure `import Creator` is done in the sub-processes, and not in the main one.
    """

    now = datetime.datetime.now()
    LOG.debug(f"Task {what}({args},{kwargs}) starting")
    LOG.error(f"✅✅✅Task {what}({args},{kwargs}) starting")
    LOG.error('')
    LOG.error('')
    LOG.error('')
    LOG.error('')

    from anemoi.datasets.create import Creator

    if "trace" in options:
        enable_trace(options["trace"])
        options.pop("trace")
    
    if "version" in options:
        options.pop("version")

    if "debug" in options:
        options.pop("debug")

    c = Creator(**options)
    result = getattr(c, what.replace('-','_'))()

    LOG.debug(f"Task {what}({args},{kwargs}) completed ({datetime.datetime.now()-now})")
    return result


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
        group = command_parser.add_mutually_exclusive_group()
        group.add_argument("--threads", help="Use `n` parallel thread workers.", type=int, default=0)
        group.add_argument("--processes", help="Use `n` parallel process workers.", type=int, default=0)
        command_parser.add_argument("--trace", action="store_true")

    def run(self, args):
        now = time.time()
        if args.threads + args.processes:
            self.parallel_create(args)
        else:
            self.serial_create(args)
        LOG.info(f"Create completed in {seconds_to_human(time.time()-now)}")

    def serial_create(self, args):
        from anemoi.datasets.create import Creator

        options = vars(args)
        options.pop("command")
        options.pop("threads")
        options.pop("processes")
        options.pop("trace")
        options.pop("debug")
        options.pop("version")
        c = Creator(**options)
        c.create()

    def parallel_create(self, args):
        """Some modules, like fsspec do not work well with fork()
        Other modules may not be thread safe. So we implement
        parallel loadining using multiprocessing before any
        of the modules are imported.
        """

        parallel = args.threads + args.processes
        args.use_threads = args.threads > 0

        if args.use_threads:
            ExecutorClass = ThreadPoolExecutor
        else:
            ExecutorClass = ProcessPoolExecutor

        options = vars(args)
        options.pop("command")
        options.pop("threads")
        options.pop("processes")

        with ExecutorClass(max_workers=1) as executor:
            total = executor.submit(task, "init", options).result()

        futures = []

        with ExecutorClass(max_workers=parallel) as executor:
            for n in range(total):
                futures.append(executor.submit(task, "load", options, parts=f"{n+1}/{total}"))

            for future in tqdm.tqdm(
                as_completed(futures), desc="Loading", total=len(futures), colour="green", position=parallel + 1
            ):
                future.result()

        with ExecutorClass(max_workers=1) as executor:
            executor.submit(task, "statistics", options).result()
            executor.submit(task, "additions", options).result()
            executor.submit(task, "cleanup", options).result()
            executor.submit(task, "verify", options).result()


command = Create
