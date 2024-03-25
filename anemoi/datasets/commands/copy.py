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

anemoi-datasets copy --source aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v3.zarr/
    --target s3://ml-datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v3.zarr
zinfo https://object-store.os-api.cci1.ecmwf.int/
    ml-datasets/stable/aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v3.zarr
"""


class CopyMixin:
    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument("--transfers", type=int, default=8)
        command_parser.add_argument("--block-size", type=int, default=100)
        command_parser.add_argument("--overwrite", action="store_true")
        command_parser.add_argument("--progress", action="store_true")
        command_parser.add_argument("--nested", action="store_true", help="Use ZARR's nested directpry backend.")
        command_parser.add_argument(
            "--rechunk",
            nargs="+",
            help="Rechunk given array.",
            metavar="array=i,j,k,l",
        )
        command_parser.add_argument("source")
        command_parser.add_argument("target")

    def _store(self, path, nested=False):
        if nested:
            import zarr

            return zarr.storage.NestedDirectoryStore(path)
        return path

    def copy_chunk(self, n, m, source, target, block_size, _copy, progress):
        if _copy[n:m].all():
            LOG.info(f"Skipping {n} to {m}")
            return None

        for i in tqdm.tqdm(
            range(n, m),
            desc=f"Copying {n} to {m}",
            leave=False,
            disable=not isatty and not progress,
        ):
            target[i] = source[i]
        return slice(n, m)

    def copy_data(self, source, target, transfers, block_size, _copy, progress, rechunking):
        LOG.info("Copying data")
        source_data = source["data"]

        chunks = list(source_data.chunks)
        if "data" in rechunking:
            assert len(chunks) == len(rechunking["data"]), (chunks, rechunking["data"])
            for i, c in enumerate(rechunking["data"]):
                if c != -1:
                    chunks[i] = c

        target_data = (
            target["data"]
            if "data" in target
            else target.create_dataset(
                "data",
                shape=source_data.shape,
                chunks=chunks,
                dtype=source_data.dtype,
            )
        )

        executor = ThreadPoolExecutor(max_workers=transfers)
        tasks = []
        n = 0
        while n < target_data.shape[0]:
            tasks.append(
                executor.submit(
                    self.copy_chunk,
                    n,
                    min(n + block_size, target_data.shape[0]),
                    source_data,
                    target_data,
                    block_size,
                    _copy,
                    progress,
                )
            )
            n += block_size

        for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), smoothing=0):
            copied = future.result()
            if copied is not None:
                _copy[copied] = True
                target["_copy"] = _copy

        target["_copy"] = _copy

        LOG.info("Copied data")

    def copy_array(self, name, source, target, transfers, block_size, _copy, progress, rechunking):
        for k, v in source.attrs.items():
            target.attrs[k] = v

        if name == "_copy":
            return

        if name == "data":
            self.copy_data(source, target, transfers, block_size, _copy, progress, rechunking)
            return

        LOG.info(f"Copying {name}")
        target[name] = source[name]
        LOG.info(f"Copied {name}")

    def copy_group(self, source, target, transfers, block_size, _copy, progress, rechunking):
        import zarr

        for k, v in source.attrs.items():
            target.attrs[k] = v

        for name in sorted(source.keys()):
            if isinstance(source[name], zarr.hierarchy.Group):
                group = target[name] if name in target else target.create_group(name)
                self.copy_group(
                    source[name],
                    group,
                    transfers,
                    block_size,
                    _copy,
                    progress,
                    rechunking,
                )
            else:
                self.copy_array(
                    name,
                    source,
                    target,
                    transfers,
                    block_size,
                    _copy,
                    progress,
                    rechunking,
                )

    def copy(self, source, target, transfers, block_size, progress, rechunking):
        import zarr

        if "_copy" not in target:
            target["_copy"] = zarr.zeros(
                source["data"].shape[0],
                dtype=bool,
            )
        _copy = target["_copy"]
        _copy_np = _copy[:]

        self.copy_group(source, target, transfers, block_size, _copy_np, progress, rechunking)
        del target["_copy"]

    def run(self, args):
        import zarr

        # base, ext = os.path.splitext(os.path.basename(args.source))
        # assert ext == ".zarr", ext
        # assert "." not in base, base
        LOG.info(f"Copying {args.source} to {args.target}")

        rechunking = {}
        if args.rechunk:
            for r in args.rechunk:
                k, v = r.split("=")
                if k != "data":
                    raise ValueError(f"Only rechunking data is supported: {k}")
                values = v.split(",")
                values = [-1 if x == "" else x for x in values]
                values = tuple(int(x) for x in values)
                rechunking[k] = values
            for k, v in rechunking.items():
                LOG.info(f"Rechunking {k} to {v}")

        try:
            target = zarr.open(self._store(args.target, args.nested), mode="r")
            if "_copy" in target:
                done = sum(1 if x else 0 for x in target["_copy"])
                todo = len(target["_copy"])
                LOG.info(
                    "Resuming copy, done %s out or %s, %s%%",
                    done,
                    todo,
                    int(done / todo * 100 + 0.5),
                )
            elif "sums" in target and "data" in target:  # sums is copied last
                LOG.error("Target already exists")
                return
        except ValueError as e:
            LOG.info(f"Target does not exist: {e}")
            pass

        source = zarr.open(self._store(args.source), mode="r")
        if args.overwrite:
            target = zarr.open(self._store(args.target, args.nested), mode="w")
        else:
            try:
                target = zarr.open(self._store(args.target, args.nested), mode="w+")
            except ValueError:
                target = zarr.open(self._store(args.target, args.nested), mode="w")
        self.copy(source, target, args.transfers, args.block_size, args.progress, rechunking)


class Copy(CopyMixin, Command):
    pass


command = Copy
