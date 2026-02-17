# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import itertools
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

import numpy as np
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


def compute_concatenated_shards(concatenate_dims: list[int], shape: tuple, chunks: tuple) -> tuple:
    """Compute shard sizes by expanding specified dimensions to the full shape extent.

    Parameters
    ----------
    concatenate_dims : list[int]
        Dimension indices along which to concatenate chunks into shards.
    shape : tuple
        Full array shape.
    chunks : tuple
        Chunk sizes.
    """
    shards = list(chunks)
    for dim in concatenate_dims:
        if dim < 0 or dim >= len(shape):
            raise ValueError(f"Concatenate dimension {dim} out of range for {len(shape)}-D array.")
        shards[dim] = shape[dim]
    return tuple(shards)


def _morton_order(shape: tuple):
    """Generate coordinates in morton (Z-order) order for a given grid shape.

    Parameters
    ----------
    shape : tuple
        Grid dimensions.
    """
    bits = tuple(math.ceil(math.log2(max(c, 2))) for c in shape)
    total = 1 << sum(bits)  # 2^(sum of all bits) to cover all morton indices
    max_bits = max(bits)
    for z in range(total):
        coords = [0] * len(shape)
        input_bit = 0
        for coord_bit in range(max_bits):
            for dim in range(len(shape)):
                if coord_bit < bits[dim]:
                    bit = (z >> input_bit) & 1
                    coords[dim] |= bit << coord_bit
                    input_bit += 1
        coords = tuple(coords)
        if all(c < s for c, s in zip(coords, shape)):
            yield coords


class StreamingShardCopier:
    """Copy data into zarr 3 sharded arrays by streaming chunks to disk.

    Bypasses zarr's write path (which reads the full shard into memory on
    every partial write) by constructing shard files directly according to
    the zarr v3 sharding specification (ZEP 2).

    Only 1 chunk + the shard index are held in memory at a time.
    Local filesystem targets only.

    Parameters
    ----------
    source_array : Any
        Source zarr array to read chunks from.
    target_group : Any
        Target zarr group to create the sharded array in.
    chunks : tuple
        Inner chunk sizes for the target array.
    shards : tuple
        Shard sizes for the target array.
    target_store_path : str
        Filesystem path to the target zarr store root.
    """

    MAX_UINT_64 = 2**64 - 1

    def __init__(self, source_array, target_group, chunks, shards, target_store_path):
        import numcodecs

        self.source = source_array
        self.shape = tuple(source_array.shape)
        self.dtype = source_array.dtype
        self.chunks = chunks
        self.shards = shards
        self.target_store_path = str(target_store_path)

        self.chunks_per_shard = tuple(s // c for s, c in zip(self.shards, self.chunks))
        self.shard_grid = tuple(-(-sh // sd) for sh, sd in zip(self.shape, self.shards))  # ceil division

        # Create the target array metadata (zarr.json) via zarr
        if "data" not in target_group:
            self.target_array = target_group.create_array(
                "data",
                shape=self.shape,
                chunks=chunks,
                shards=shards,
                dtype=self.dtype,
                fill_value=source_array.fill_value,
            )
        else:
            self.target_array = target_group["data"]

        # Default inner codec: Zstd level 0 (matches zarr's default)
        self._compressor = numcodecs.Zstd(level=0)

    def _encode_chunk(self, data: np.ndarray) -> bytes:
        """Encode a single chunk: little-endian bytes + zstd compression."""
        raw = np.ascontiguousarray(data).astype(self.dtype.newbyteorder("<")).tobytes()
        return self._compressor.encode(raw)

    def _encode_index(self, index_array: np.ndarray) -> bytes:
        """Encode the shard index with BytesCodec + Crc32cCodec."""
        import google_crc32c

        raw = index_array.tobytes()
        crc = google_crc32c.value(raw)
        return raw + np.array([crc], dtype=np.uint32).tobytes()

    def _shard_file_path(self, shard_grid_pos: tuple) -> str:
        """Return filesystem path for the shard at given grid position."""
        parts = "/".join(str(i) for i in shard_grid_pos)
        return os.path.join(self.target_store_path, "data", "c", parts)

    def _chunk_global_slices(self, shard_pos: tuple, local_coords: tuple) -> tuple[slice, ...]:
        """Compute the global array slices for a chunk within a shard."""
        slices = []
        for dim in range(len(self.shape)):
            start = shard_pos[dim] * self.shards[dim] + local_coords[dim] * self.chunks[dim]
            end = min(start + self.chunks[dim], self.shape[dim])
            slices.append(slice(start, end))
        return tuple(slices)

    def copy_all(self) -> None:
        """Stream all data from source to target, one chunk at a time."""
        total_shards = 1
        for g in self.shard_grid:
            total_shards *= g
        total_chunks = total_shards
        for c in self.chunks_per_shard:
            total_chunks *= c

        LOG.info(
            f"Streaming copy: {total_shards} shard(s), "
            f"{total_chunks} chunk(s), "
            f"chunks_per_shard={self.chunks_per_shard}, "
            f"shard_grid={self.shard_grid}"
        )

        chunk_count = 0
        with tqdm.tqdm(total=total_chunks, desc="Streaming chunks") as pbar:
            for shard_pos in itertools.product(*[range(g) for g in self.shard_grid]):
                written = self._write_one_shard(shard_pos)
                chunk_count += written
                pbar.update(written)

        LOG.info(f"Wrote {chunk_count} chunks into {total_shards} shard file(s)")

    def _write_one_shard(self, shard_pos: tuple) -> int:
        """Build one shard file by streaming chunks sequentially."""
        index = np.full(self.chunks_per_shard + (2,), self.MAX_UINT_64, dtype="<u8")

        path = self._shard_file_path(shard_pos)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        chunk_count = 0
        with open(path, "wb") as f:
            offset = 0
            for local_coords in _morton_order(self.chunks_per_shard):
                global_slices = self._chunk_global_slices(shard_pos, local_coords)
                chunk_data = self.source[global_slices]

                encoded = self._encode_chunk(chunk_data)
                f.write(encoded)

                index[local_coords + (0,)] = offset
                index[local_coords + (1,)] = len(encoded)
                offset += len(encoded)
                chunk_count += 1

            f.write(self._encode_index(index))

        return chunk_count


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
    rechunk : str
        Rechunk size for the target data array.
    reshard : str
        Reshard size for the target data array.
    concatenate : list[int] or None
        Dimension indices along which to concatenate chunks into shards.
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
        concatenate: str = None,
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
        concatenate : str
            Concatenate chunks into shards along given dimension(s).
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
        self.concatenate = [int(d) for d in concatenate.split(",")] if concatenate else None

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

        if self.concatenate is not None:
            assert self.transfers == 1, f"Concatenation requires single-threaded writes, got {self.transfers}"

            chunks = self.parse_rechunking(self.rechunking, source_data) if self.rechunking else source_data.chunks
            shards = compute_concatenated_shards(self.concatenate, source_data.shape, chunks)

            # Validate shard/chunk divisibility
            for s, c in zip(shards, chunks):
                if s % c != 0:
                    raise ValueError(f"Shard size {s} is not divisible by chunk size {c}.")

            LOG.info(f"Concatenating along dims {self.concatenate}: shards={shards}, chunks={chunks}")

            copier = StreamingShardCopier(
                source_array=source_data,
                target_group=target,
                chunks=chunks,
                shards=shards,
                target_store_path=self.target,
            )
            copier.copy_all()

            end = time.time()
            LOG.info(f"Copied data in {end - start:.2f} seconds")
            return

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
                    raise ValueError(f"Shard size {shard} is not a multiple of chunk size {chunk}.")
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
            command_parser.add_argument(
                "--concatenate",
                help="Concatenate chunks into shards along given dimension(s). "
                "E.g. '0' expands shard to full shape along dim 0. "
                "Comma-separated for multiple dims: '0,1'. "
                "Requires --transfers 1 (sequential writes). "
                "Local target store only.",
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

        concatenate_requested = getattr(args, "concatenate", None)

        if concatenate_requested and getattr(args, "reshard", None):
            raise ValueError("Cannot use --concatenate and --reshard together.")

        if concatenate_requested and args.transfers != 1:
            raise ValueError(
                f"--concatenate requires --transfers 1 (sequential writes), got --transfers {args.transfers}."
            )

        if concatenate_requested and args.resume:
            raise ValueError("--concatenate does not support --resume.")

        if zarr_version >= 3:
            reshaping_requested = args.rechunk or args.reshard or concatenate_requested
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
