# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import tqdm

from . import Command

LOG = logging.getLogger(__name__)

try:
    isatty = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
except AttributeError:
    isatty = False

"""

~/.aws/credentials

[default]
endpoint_url = https://object-store.os-api.cci1.ecmwf.int
aws_access_key_id=xxx
aws_secret_access_key=xxxx

Then:

anemoi-datasets copy aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v3.zarr/
    s3://ml-datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v3.zarr

zinfo https://object-store.os-api.cci1.ecmwf.int/
    ml-datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v3.zarr
"""


class Copier:
    def __init__(self, source, target, transfers, block_size, overwrite, resume, progress, nested, rechunk, **kwargs):
        self.source = source
        self.target = target
        self.transfers = transfers
        self.block_size = block_size
        self.overwrite = overwrite
        self.resume = resume
        self.progress = progress
        self.nested = nested
        self.rechunk = rechunk

        self.rechunking = rechunk.split(",") if rechunk else []

    def _store(self, path, nested=False):
        if nested:
            import zarr

            return zarr.storage.NestedDirectoryStore(path)
        return path

    def copy_chunk(self, n, m, source, target, _copy, progress):
        if _copy[n:m].all():
            LOG.info(f"Skipping {n} to {m}")
            return None

        if self.block_size % self.data_chunks[0] == 0:
            target[slice(n, m)] = source[slice(n, m)]
        else:
            LOG.warning(
                f"Block size ({self.block_size}) is not a multiple of target chunk size ({self.data_chunks[0]}). Slow copy expected."
            )
            if self.transfers > 1:
                # race condition, different threads might copy the same data to the same chunk
                raise NotImplementedError(
                    "Block size is not a multiple of target chunk size. Parallel copy not supported."
                )
            for i in tqdm.tqdm(
                range(n, m),
                desc=f"Copying {n} to {m}",
                leave=False,
                disable=not isatty and not progress,
            ):
                target[i] = source[i]

        return slice(n, m)

    def parse_rechunking(self, rechunking, source_data):
        shape = source_data.shape
        chunks = list(source_data.chunks)
        for i, c in enumerate(rechunking):
            if not c:
                continue
            elif c == "full":
                chunks[i] = shape[i]
            c = int(c)
            c = min(c, shape[i])
            chunks[i] = c
        chunks = tuple(chunks)

        if chunks != source_data.chunks:
            LOG.info(f"Rechunking data from {source_data.chunks} to {chunks}")
            # if self.transfers > 1:
            #    raise NotImplementedError("Rechunking with multiple transfers is not implemented")
        return chunks

    def copy_data(self, source, target, _copy, progress):
        LOG.info("Copying data")
        source_data = source["data"]

        self.data_chunks = self.parse_rechunking(self.rechunking, source_data)

        target_data = (
            target["data"]
            if "data" in target
            else target.create_dataset(
                "data",
                shape=source_data.shape,
                chunks=self.data_chunks,
                dtype=source_data.dtype,
            )
        )

        executor = ThreadPoolExecutor(max_workers=self.transfers)
        tasks = []
        n = 0
        while n < target_data.shape[0]:
            tasks.append(
                executor.submit(
                    self.copy_chunk,
                    n,
                    min(n + self.block_size, target_data.shape[0]),
                    source_data,
                    target_data,
                    _copy,
                    progress,
                )
            )
            n += self.block_size

        for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), smoothing=0):
            copied = future.result()
            if copied is not None:
                _copy[copied] = True
                target["_copy"] = _copy

        target["_copy"] = _copy

        LOG.info("Copied data")

    def copy_array(self, name, source, target, _copy, progress):
        for k, v in source.attrs.items():
            target.attrs[k] = v

        if name == "_copy":
            return

        if name == "data":
            self.copy_data(source, target, _copy, progress)
            return

        LOG.info(f"Copying {name}")
        target[name] = source[name]
        LOG.info(f"Copied {name}")

    def copy_group(self, source, target, _copy, progress):
        import zarr

        for k, v in source.attrs.items():
            target.attrs[k] = v

        for name in sorted(source.keys()):
            if isinstance(source[name], zarr.hierarchy.Group):
                group = target[name] if name in target else target.create_group(name)
                self.copy_group(
                    source[name],
                    group,
                    _copy,
                    progress,
                )
            else:
                self.copy_array(
                    name,
                    source,
                    target,
                    _copy,
                    progress,
                )

    def copy(self, source, target, progress):
        import zarr

        if "_copy" not in target:
            target["_copy"] = zarr.zeros(
                source["data"].shape[0],
                dtype=bool,
            )
        _copy = target["_copy"]
        _copy_np = _copy[:]

        self.copy_group(source, target, _copy_np, progress)
        del target["_copy"]

    def run(self):
        import zarr

        # base, ext = os.path.splitext(os.path.basename(args.source))
        # assert ext == ".zarr", ext
        # assert "." not in base, base
        LOG.info(f"Copying {self.source} to {self.target}")

        def target_exists():
            try:
                zarr.open(self._store(self.target), mode="r")
                return True
            except ValueError:
                return False

        def target_finished():
            target = zarr.open(self._store(self.target), mode="r")
            if "_copy" in target:
                done = sum(1 if x else 0 for x in target["_copy"])
                todo = len(target["_copy"])
                LOG.info(
                    "Resuming copy, done %s out or %s, %s%%",
                    done,
                    todo,
                    int(done / todo * 100 + 0.5),
                )
                return False
            elif "sums" in target and "data" in target:  # sums is copied last
                return True
            return False

        def open_target():

            if not target_exists():
                return zarr.open(self._store(self.target, self.nested), mode="w")

            if self.overwrite:
                LOG.error("Target already exists, overwriting.")
                return zarr.open(self._store(self.target, self.nested), mode="w")

            if self.resume:
                if target_finished():
                    LOG.error("Target already exists and is finished.")
                    sys.exit(0)

                LOG.error("Target already exists, resuming copy.")
                return zarr.open(self._store(self.target, self.nested), mode="w+")

            LOG.error("Target already exists, use either --overwrite or --resume.")
            sys.exit(1)

        target = open_target()

        assert target is not None, target

        source = zarr.open(self._store(self.source), mode="r")
        self.copy(source, target, self.progress)


class CopyMixin:
    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        group = command_parser.add_mutually_exclusive_group()
        group.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing dataset. This will delete the target dataset if it already exists. Cannot be used with --resume.",
        )
        group.add_argument(
            "--resume", action="store_true", help="Resume copying an existing dataset. Cannot be used with --overwrite."
        )
        command_parser.add_argument("--transfers", type=int, default=8, help="Number of parallel transfers.")
        command_parser.add_argument(
            "--progress", action="store_true", help="Force show progress bar, even if not in an interactive shell."
        )
        command_parser.add_argument("--nested", action="store_true", help="Use ZARR's nested directpry backend.")
        command_parser.add_argument(
            "--rechunk", help="Rechunk the target data array. Rechunk size should be a diviser of the block size."
        )
        command_parser.add_argument(
            "--block-size",
            type=int,
            default=100,
            help="For optimisation purposes, data is transfered by blocks. Default is 100.",
        )
        command_parser.add_argument("source", help="Source location.")
        command_parser.add_argument("target", help="Target location.")

    def run(self, args):
        Copier(**vars(args)).run()


class Copy(CopyMixin, Command):
    """Copy a dataset from one location to another."""


command = Copy
