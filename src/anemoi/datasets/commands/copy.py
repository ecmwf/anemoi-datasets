# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

import tqdm
from anemoi.utils.remote import Transfer
from anemoi.utils.remote import TransferMethodNotImplementedError

from anemoi.datasets.data.stores import name_to_zarr_store

from ..compat import ZarrFileNotFoundError
from ..compat import zarr_append_mode
from ..compat import zarr_private_files
from ..compat import zarr_version
from . import Command

LOG = logging.getLogger(__name__)


class ZarrCopier:
    """Class to handle copying of Zarr datasets.

    Attributes
    ----------
    source : str
        Source location of the dataset.
    target : str
        Target location of the dataset.
    transfers : int
        Number of parallel transfers.
    block_size : int
        Size of data blocks to transfer.
    overwrite : bool
        Flag to overwrite existing dataset.
    resume : bool
        Flag to resume copying an existing dataset.
    verbosity : int
        Verbosity level of logging.
    nested : bool
        Flag to use ZARR's nested directory backend.
    rechunk : str
        Rechunk size for the target data array.
    reshard : str
        Reshard size for the target data array.
    """

    def __init__(
        self,
        source: str,
        target: str,
        transfers: int,
        block_size: int,
        overwrite: bool,
        resume: bool,
        verbosity: int,
        rechunk: str,
        reshard: str = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ZarrCopier.

        Parameters
        ----------
        source : str
            Source location of the dataset.
        target : str
            Target location of the dataset.
        transfers : int
            Number of parallel transfers.
        block_size : int
            Size of data blocks to transfer.
        overwrite : bool
            Flag to overwrite existing dataset.
        resume : bool
            Flag to resume copying an existing dataset.
        verbosity : int
            Verbosity level of logging.
        rechunk : str
            Rechunk size for the target data array.
        reshard : str
            Reshard size for the target data array.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.source = name_to_zarr_store(source)
        self.target = name_to_zarr_store(target)
        self.transfers = transfers
        self.block_size = block_size
        self.overwrite = overwrite
        self.resume = resume
        self.verbosity = verbosity

        self.rechunking = rechunk.split(",") if rechunk else []
        self.resharding = reshard.split(",") if reshard else []

    def copy_chunk(self, n: int, m: int, source: Any, target: Any, _copy: Any, verbosity: int) -> slice | None:
        """Copy a chunk of data from source to target.

        Parameters
        ----------
        n : int
            Start index of the chunk.
        m : int
            End index of the chunk.
        source : Any
            Source data.
        target : Any
            Target data.
        _copy : Any
            Copy status array.
        verbosity : int
            Verbosity level of logging.

        Returns
        -------
        slice or None
            Slice of copied data or None if skipped.
        """
        if _copy[n:m].all():
            LOG.info(f"Skipping {n} to {m}")
            return None

        target[slice(n, m)] = source[slice(n, m)]

        return slice(n, m)

    def _parse_reshaping(self, new, old, shape) -> tuple:
        if old is not None:
            old = list(old)

        if new is None:
            return old

        result = [s for s in (shape if old is None else old)]

        for i, c in enumerate(new):
            if c in ("full", "-1", ""):
                continue
            c = int(c)
            c = min(c, shape[i])
            result[i] = c

        result = tuple(result)
        return result

    def parse_rechunking(self, rechunking: list[str], source_data: Any) -> tuple:
        chunks = self._parse_reshaping(new=rechunking, old=source_data.chunks, shape=source_data.shape)

        if chunks != source_data.chunks:
            LOG.info(f"Rechunking data from {source_data.chunks} to {chunks}")

        return chunks

    def parse_resharding(self, resharding: list[str], source_data: Any) -> tuple:
        shards = self._parse_reshaping(new=resharding, old=source_data.shards, shape=source_data.shape)

        if shards != source_data.shards:
            LOG.info(f"Resharding data from {source_data.shards} to {shards} (shape is {source_data.shape})")

        return shards

    def copy_data(self, source: Any, target: Any, _copy: Any, verbosity: int) -> None:
        """Copy data from source to target.

        Parameters
        ----------
        source : Any
            Source data.
        target : Any
            Target data.
        _copy : Any
            Copy status array.
        verbosity : int
            Verbosity level of logging.
        """
        LOG.info("Copying data")
        source_data = source["data"]
        start = time.time()

        extra = {}
        if self.rechunking:
            extra["chunks"] = self.parse_rechunking(self.rechunking, source_data)

        if self.resharding:
            extra["shards"] = self.parse_resharding(self.resharding, source_data)
            extra["chunks"] = (
                self.parse_rechunking(self.rechunking, source_data) if self.rechunking else source_data.chunks
            )
            ratio = []
            for shard, chunk in zip(extra["shards"], extra["chunks"]):
                if shard % chunk != 0:
                    raise ValueError(f"Chunk size {chunk} is not a multiple of shard size {shard}.")
                ratio.append(shard // chunk)

            LOG.info(f"Shards for target data array: {extra['shards']} (ratio={ratio})")

        LOG.info(f"Chunks: source={source_data.chunks}")
        if zarr_version >= 3:
            LOG.info(f"Shards: source={source_data.shards}")

        if extra:
            LOG.info(f"Using extra parameters for target data array: {extra}")

        if "data" in target:
            target_data = target["data"]
            if extra:
                LOG.warning("Target data array already exists, ignoring resharding/rechunking parameters.")
                LOG.warning(f"Existing target data array chunks: {target_data.chunks}")
                if zarr_version >= 3:
                    LOG.warning(f"Existing target data array shards: {target_data.shards}")
        else:
            extra.setdefault("chunks", source_data.chunks)
            target_data = target.create_array(
                "data",
                shape=source_data.shape,
                dtype=source_data.dtype,
                fill_value=source_data.fill_value,
                **extra,
            )

        size = 1
        size = target_data.chunks[0] if target_data.chunks else size
        if zarr_version >= 3:
            size = target_data.shards[0] if target_data.shards else size

        block_size = self.block_size

        block_size = (block_size // size) * size
        if block_size < size:
            block_size = size

        if block_size != self.block_size:
            LOG.info(
                f"Adjusted block size from {self.block_size} to {block_size} to be multiple of chunk/shard size {size} {target_data.chunks}."
            )
            self.block_size = block_size

        LOG.info(f"Using block size {self.block_size}, parallel transfers {self.transfers}")
        self.block_size = block_size

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

        end = time.time()
        LOG.info(f"Copied data in {end - start:.2f} seconds")

    def copy_array(self, name: str, source: Any, target: Any, _copy: Any, verbosity: int) -> None:
        """Copy an array from source to target.

        Parameters
        ----------
        name : str
            Name of the array.
        source : Any
            Source data.
        target : Any
            Target data.
        _copy : Any
            Copy status array.
        verbosity : int
            Verbosity level of logging.
        """
        for k, v in source.attrs.items():
            target.attrs[k] = v

        if name == "_copy":
            return

        if name == "data":
            self.copy_data(source, target, _copy, verbosity)
            return

        LOG.info(f"Copying {name} {source[name].shape}")
        data = source[name][...]
        if name in target:
            del target[name]
        target.create_dataset(name, data=data, shape=data.shape)
        LOG.info(f"Copied {name}")

    def children(self, group):
        children = list(group.keys())
        # https://github.com/zarr-developers/zarr-python/issues/3575
        children = [k for k in children if k != ""]
        children = [k for k in children if k not in zarr_private_files]
        children = sorted(children)
        return children

    def copy_group(self, source: Any, target: Any, _copy: Any, verbosity: int) -> None:
        """Copy a group from source to target.

        Parameters
        ----------
        source : Any
            Source data.
        target : Any
            Target data.
        _copy : Any
            Copy status array.
        verbosity : int
            Verbosity level of logging.
        """
        import zarr

        if self.verbosity > 0:
            LOG.info(f"Copying group {source} to {target}")

        for k, v in source.attrs.items():
            if self.verbosity > 1:
                import textwrap

                LOG.info(f"Copying attribute {k} = {textwrap.shorten(str(v), 40)}")
            target.attrs[k] = v

        source_keys = self.children(source)

        if not source_keys:
            raise ValueError(f"Source group {source} is empty.")

        if self.verbosity > 1:
            LOG.info(f"Keys {source_keys}")

        for name in source_keys:

            if isinstance(source[name], zarr.Group):
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

    def copy(self, source: Any, target: Any, verbosity: int) -> None:
        """Copy the entire dataset from source to target.

        Parameters
        ----------
        source : Any
            Source data.
        target : Any
            Target data.
        verbosity : int
            Verbosity level of logging.
        """

        if "_copy" not in target:
            target.create_dataset(
                "_copy",
                shape=(source["data"].shape[0],),
                dtype=bool,
            )
        _copy = target["_copy"]
        _copy_np = _copy[:]

        if self.verbosity > 1:
            import numpy as np

            LOG.info(f"copy {np.sum(_copy_np)} of {len(_copy_np)}")

        self.copy_group(source, target, _copy_np, verbosity)
        del target["_copy"]

    def run(self) -> None:
        """Execute the copy operation."""
        import zarr

        # base, ext = os.path.splitext(os.path.basename(args.source))
        # assert ext == ".zarr", ext
        # assert "." not in base, base
        LOG.info(f"Copying {self.source} to {self.target}")
        LOG.info(f"Zarr version {zarr.__version__}")

        def target_exists() -> bool:
            try:
                zarr.open(self.target, mode="r")
                return True
            except ZarrFileNotFoundError:
                return False

        def target_finished() -> bool:
            target = zarr.open(self.target, mode="r")
            source = zarr.open(self.source, mode="r")
            last_key = list(self.children(source))[-1]
            if "_copy" in target:
                done = sum(1 if x else 0 for x in target["_copy"])
                todo = target["_copy"].shape[0]
                LOG.info(
                    "Resuming copy, done %s out or %s, %s%%",
                    done,
                    todo,
                    int(done / todo * 100 + 0.5),
                )
                return False
            elif last_key in target and "data" in target:
                return True

            return False

        def open_target() -> Any:

            if not target_exists():
                return zarr.open(self.target, mode="w")

            if self.overwrite:
                LOG.error("Target already exists, overwriting.")
                return zarr.open(self.target, mode="w")

            if self.resume:
                if target_finished():
                    LOG.error("Target already exists and is finished.")
                    sys.exit(0)

                LOG.error("Target already exists, resuming copy.")
                return zarr.open(self.target, mode=zarr_append_mode)

            LOG.error("Target already exists, use either --overwrite or --resume.")
            sys.exit(1)

        source = zarr.open(self.source, mode="r")
        if self.verbosity > 0:
            LOG.info(f"Open source: {self.source}")

        if self.verbosity > 0:
            LOG.info(f"Open target: {self.target}")

        target = open_target()

        assert target is not None, target

        self.copy(source, target, self.verbosity)


class CopyMixin:
    """Mixin class for adding copy command arguments and running the copy operation."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            Command parser object.
        """
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

        command_parser.add_argument(
            "--block-size",
            type=int,
            default=100,
            help="For optimisation purposes, data is transfered by blocks. Default is 100.",
        )

        command_parser.add_argument(
            "--rechunk",
            help="Rechunk the target data array. This option will adjust --block-size to that it is divisible by the rechunk size.",
        )

        if zarr_version >= 3:
            command_parser.add_argument(
                "--reshard",
                help="Reshard the target data array. This option will adjust --block-size to that it is divisible by the reshard size.",
            )

        command_parser.add_argument("source", help="Source location.")
        command_parser.add_argument("target", help="Target location.")

    def run(self, args: Any) -> None:
        """Run the copy command with the provided arguments.

        Parameters
        ----------
        args : Any
            Command arguments.
        """
        if args.source == args.target:
            raise ValueError("Source and target are the same.")

        if args.overwrite and args.resume:
            raise ValueError("Cannot use --overwrite and --resume together.")

        if zarr_version >= 3:
            reshaping_requested = args.rechunk or args.reshard
        else:
            reshaping_requested = args.rechunk

        if not reshaping_requested:
            # rechunking is only supported for ZARR datasets, it is implemented in this package
            try:
                if args.source.startswith("s3://") and not args.source.endswith("/"):
                    args.source = args.source + "/"
                copier = Transfer(
                    source=args.source,
                    target=args.target,
                    overwrite=args.overwrite,
                    resume=args.resume,
                    verbosity=args.verbosity,
                    threads=args.transfers,
                )
                copier.run()
                return
            except TransferMethodNotImplementedError:
                # DataTransfer relies on anemoi-utils which is agnostic to the source and target format
                # it transfers file and folders, ignoring that it is zarr data
                # if it is not implemented, we fallback to the ZarrCopier
                pass

        copier = ZarrCopier(**vars(args))
        copier.run()
        return


class Copy(CopyMixin, Command):
    """Copy a dataset from one location to another."""


command = Copy
