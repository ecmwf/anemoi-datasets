# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any
from typing import Optional

import tqdm
from anemoi.utils.remote import Transfer
from anemoi.utils.remote import TransferMethodNotImplementedError

from anemoi.datasets.check import check_zarr

from . import Command

LOG = logging.getLogger(__name__)

try:
    isatty = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
except AttributeError:
    isatty = False


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
        nested: bool,
        rechunk: str,
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
        nested : bool
            Flag to use ZARR's nested directory backend.
        rechunk : str
            Rechunk size for the target data array.
        **kwargs : Any
            Additional keyword arguments.
        """
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

        source_is_ssh = self.source.startswith("ssh://")
        target_is_ssh = self.target.startswith("ssh://")

        if source_is_ssh or target_is_ssh:
            if self.rechunk:
                raise NotImplementedError("Rechunking with SSH not implemented.")
            assert NotImplementedError("SSH not implemented.")

    def _store(self, path: str, nested: bool = False) -> Any:
        """Get the storage path.

        Parameters
        ----------
        path : str
            Path to the storage.
        nested : bool, optional
            Flag to use nested directory storage.

        Returns
        -------
        Any
            Storage path.
        """
        if nested:
            import zarr

            return zarr.storage.NestedDirectoryStore(path)
        return path

    def copy_chunk(self, n: int, m: int, source: Any, target: Any, _copy: Any, verbosity: int) -> Optional[slice]:
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

    def parse_rechunking(self, rechunking: list[str], source_data: Any) -> tuple:
        """Parse the rechunking configuration.

        Parameters
        ----------
        rechunking : list of str
            List of rechunk sizes.
        source_data : Any
            Source data.

        Returns
        -------
        tuple
            Parsed chunk sizes.
        """
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

        LOG.info(f"Copying {name}")
        target[name] = source[name]
        LOG.info(f"Copied {name}")

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

        source_keys = list(source.keys())

        if not source_keys:
            raise ValueError(f"Source group {source} is empty.")

        if self.verbosity > 1:
            LOG.info(f"Keys {source_keys}")

        for name in sorted(source_keys):
            if name.startswith("."):
                if self.verbosity > 1:
                    LOG.info(f"Skipping {name}")
                continue

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
        import zarr

        if "_copy" not in target:
            target["_copy"] = zarr.zeros(
                source["data"].shape[0],
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

        def target_exists() -> bool:
            try:
                zarr.open(self._store(self.target), mode="r")
                return True
            except ValueError:
                return False

        def target_finished() -> bool:
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

        def open_target() -> Any:

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

        if self.verbosity > 0:
            LOG.info(f"Open target: {self.target}")

        target = open_target()

        assert target is not None, target

        if self.verbosity > 0:
            LOG.info(f"Open source: {self.source}")

        source = zarr.open(self._store(self.source), mode="r")
        # zarr.consolidate_metadata(source)

        self.copy(source, target, self.verbosity)
        if os.path.exists(self.target) and os.path.isdir(self.target):
            LOG.info(f"Checking target: {self.target}")
            check_zarr(self.target, self.verbosity)
        else:
            LOG.info(f"Target {self.target} is not a local directory, skipping check.")


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

        if not args.rechunk:
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
