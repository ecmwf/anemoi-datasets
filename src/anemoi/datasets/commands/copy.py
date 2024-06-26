# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import tqdm
from anemoi.utils.s3 import download
from anemoi.utils.s3 import upload

from . import Command

LOG = logging.getLogger(__name__)

try:
    isatty = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
except AttributeError:
    isatty = False


class S3Downloader:
    def __init__(self, source, target, transfers, overwrite, resume, verbosity, **kwargs):
        self.source = source
        self.target = target
        self.transfers = transfers
        self.overwrite = overwrite
        self.resume = resume
        self.verbosity = verbosity

    def run(self):
        if self.target == ".":
            self.target = os.path.basename(self.source)

        if self.overwrite and os.path.exists(self.target):
            LOG.info(f"Deleting {self.target}")
            shutil.rmtree(self.target)

        download(
            self.source + "/" if not self.source.endswith("/") else self.source,
            self.target,
            overwrite=self.overwrite,
            resume=self.resume,
            verbosity=self.verbosity,
            threads=self.transfers,
        )


class S3Uploader:
    def __init__(self, source, target, transfers, overwrite, resume, verbosity, **kwargs):
        self.source = source
        self.target = target
        self.transfers = transfers
        self.overwrite = overwrite
        self.resume = resume
        self.verbosity = verbosity

    def run(self):
        upload(
            self.source,
            self.target,
            overwrite=self.overwrite,
            resume=self.resume,
            verbosity=self.verbosity,
            threads=self.transfers,
        )


class DefaultCopier:
    def __init__(self, source, target, transfers, block_size, overwrite, resume, verbosity, nested, rechunk, **kwargs):
        self.source = source
        self.target = target
        self.transfers = transfers
        self.block_size = block_size
        self.overwrite = overwrite
        self.resume = resume
        self.verbosity = verbosity
        self.nested = nested
        self.rechunk = rechunk

        self.rechunking = rechunk.split(",") if rechunk else []

    def _store(self, path, nested=False):
        if nested:
            import zarr

            return zarr.storage.NestedDirectoryStore(path)
        return path

    def copy_chunk(self, n, m, source, target, _copy, verbosity):
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
                disable=not isatty and not verbosity,
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

    def copy_data(self, source, target, _copy, verbosity):
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
                fill_value=source_data.fill_value,
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
                    verbosity,
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

    def copy_array(self, name, source, target, _copy, verbosity):
        for k, v in source.attrs.items():
            target.attrs[k] = v

        if name == "_copy":
            return

        if name == "data":
            self.copy_data(source, target, _copy, verbosity)
            return

        LOG.info(f"Copying {name}")
        target[name] = source[name]
        LOG.info(f"Copied {name}")

    def copy_group(self, source, target, _copy, verbosity):
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
                    verbosity,
                )
            else:
                self.copy_array(
                    name,
                    source,
                    target,
                    _copy,
                    verbosity,
                )

    def copy(self, source, target, verbosity):
        import zarr

        if "_copy" not in target:
            target["_copy"] = zarr.zeros(
                source["data"].shape[0],
                dtype=bool,
            )
        _copy = target["_copy"]
        _copy_np = _copy[:]

        self.copy_group(source, target, _copy_np, verbosity)
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
        self.copy(source, target, self.verbosity)


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
            "--verbosity",
            type=int,
            help="Verbosity level. 0 is silent, 1 is normal, 2 is verbose.",
            default=1,
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
        if args.source == args.target:
            raise ValueError("Source and target are the same.")

        kwargs = vars(args)

        if args.overwrite and args.resume:
            raise ValueError("Cannot use --overwrite and --resume together.")

        source_in_s3 = args.source.startswith("s3://")
        target_in_s3 = args.target.startswith("s3://")

        copier = None

        if args.rechunk or (source_in_s3 and target_in_s3):
            copier = DefaultCopier(**kwargs)
        else:
            if source_in_s3:
                copier = S3Downloader(**kwargs)

            if target_in_s3:
                copier = S3Uploader(**kwargs)

        copier.run()


class Copy(CopyMixin, Command):
    """Copy a dataset from one location to another."""


command = Copy
