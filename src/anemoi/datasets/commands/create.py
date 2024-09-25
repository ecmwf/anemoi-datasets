import datetime
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import tqdm
from anemoi.utils.humanize import seconds_to_human

from . import Command

LOG = logging.getLogger(__name__)


def task(what, options, *args, **kwargs):
    """
    Make sure `import Creator` is done in the sub-processes, and not in the main one.
    """

    now = datetime.datetime.now()
    LOG.info(f"🎬 Task {what}({args},{kwargs}) starting")

    from anemoi.datasets.create import creator_factory

    options = {k: v for k, v in options.items() if v is not None}

    c = creator_factory(what.replace("-", "_"), **options)
    result = c.run()

    LOG.info(f"🏁 Task {what}({args},{kwargs}) completed ({datetime.datetime.now()-now})")
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

        options = vars(args)
        options.pop("command")
        options.pop("threads")
        options.pop("processes")

        task("init", options)
        task("load", options)
        task("finalise", options)

        task("patch", options)

        task("init_additions", options)
        task("run_additions", options)
        task("finalise_additions", options)
        task("cleanup", options)
        task("verify", options)

    def parallel_create(self, args):
        """Some modules, like fsspec do not work well with fork()
        Other modules may not be thread safe. So we implement
        parallel loadining using multiprocessing before any
        of the modules are imported.
        """

        options = vars(args)
        options.pop("command")

        threads = options.pop("threads")
        processes = options.pop("processes")

        use_threads = threads > 0
        options["use_threads"] = use_threads

        if use_threads:
            ExecutorClass = ThreadPoolExecutor
        else:
            ExecutorClass = ProcessPoolExecutor

        with ExecutorClass(max_workers=1) as executor:
            total = executor.submit(task, "init", options).result()

        futures = []

        parallel = threads + processes
        with ExecutorClass(max_workers=parallel) as executor:
            for n in range(total):
                opt = options.copy()
                opt["parts"] = f"{n+1}/{total}"
                futures.append(executor.submit(task, "load", opt))

            for future in tqdm.tqdm(as_completed(futures), desc="Loading", total=len(futures), colour="green", position=parallel + 1):  # fmt: skip
                future.result()

        with ExecutorClass(max_workers=1) as executor:
            executor.submit(task, "finalise", options).result()

        with ExecutorClass(max_workers=1) as executor:
            executor.submit(task, "init-additions", options).result()

        with ExecutorClass(max_workers=parallel) as executor:
            opt = options.copy()
            opt["parts"] = f"{n+1}/{total}"
            futures.append(executor.submit(task, "load", opt))
            for n in range(total):
                futures.append(executor.submit(task, "load-additions", opt))

            for future in tqdm.tqdm(as_completed(futures), desc="Computing additions", total=len(futures), colour="green", position=parallel + 1):  # fmt: skip
                future.result()

        with ExecutorClass(max_workers=1) as executor:
            executor.submit(task, "finalise-additions", options).result()
            executor.submit(task, "cleanup", options).result()
            executor.submit(task, "verify", options).result()


command = Create
