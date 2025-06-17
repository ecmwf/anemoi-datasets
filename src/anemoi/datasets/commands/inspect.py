# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import semantic_version
import tqdm
from anemoi.utils.humanize import bytes
from anemoi.utils.humanize import bytes_to_human
from anemoi.utils.humanize import when
from anemoi.utils.text import dotted_line
from anemoi.utils.text import progress
from anemoi.utils.text import table
from numpy.typing import NDArray

from anemoi.datasets import open_dataset
from anemoi.datasets.data.stores import open_zarr
from anemoi.datasets.data.stores import zarr_lookup

from . import Command

LOG = logging.getLogger(__name__)


def compute_directory_size(path: str) -> Union[tuple[int, int], tuple[None, None]]:
    """Compute the total size and number of files in a directory.

    Parameters
    ----------
    path : str
        The path to the directory.

    Returns
    -------
    tuple[int, int] | tuple[None, None]
        The total size in bytes and the number of files,
        or (None, None) if the path is not a directory.
    """
    if not os.path.isdir(path):
        return None, None
    size = 0
    n = 0
    for dirpath, _, filenames in tqdm.tqdm(os.walk(path), desc="Computing size", leave=False):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            size += os.path.getsize(file_path)
            n += 1
    return size, n


def local_time_bug(lon: float, date: datetime.datetime) -> float:
    """Calculate the local time bug based on longitude and date.

    Parameters
    ----------
    lon : float
        Longitude.
    date : datetime.datetime
        Date and time.

    Returns
    -------
    float
        Local time bug in hours.
    """
    delta = date - datetime.datetime(date.year, date.month, date.day)
    hours_since_midnight = delta.days + delta.seconds / 86400.0  # * 24 is missing
    return (lon / 360.0 * 24.0 + hours_since_midnight) % 24


def cos_local_time_bug(lon: float, date: datetime.datetime) -> float:
    """Calculate the cosine of the local time bug.

    Parameters
    ----------
    lon : float
        Longitude.
    date : datetime.datetime
        Date and time.

    Returns
    -------
    float
        Cosine of the local time bug.
    """
    radians = local_time_bug(lon, date) / 24 * np.pi * 2
    return np.cos(radians)


def find(config: Union[dict, list], name: str) -> Any:
    """Recursively search for a key in a nested dictionary or list.

    Parameters
    ----------
    config : dict or list
        The configuration to search.
    name : str
        The key to search for.

    Returns
    -------
    Any
        The value associated with the key, or None if not found.
    """
    if isinstance(config, dict):
        if name in config:
            return config[name]

        for k, v in config.items():
            r = find(v, name)
            if r is not None:
                return r

    if isinstance(config, list):
        for v in config:
            r = find(v, name)
            if r is not None:
                return r

    return None


class Version:
    """Represents a version of a dataset."""

    def __init__(self, path: str, zarr: Any, metadata: dict, version: semantic_version.Version) -> None:
        """Initialize the Version object.

        Parameters
        ----------
        path : str
            Path to the dataset.
        zarr : Any
            Zarr object.
        metadata : dict
            Metadata of the dataset.
        version : semantic_version.Version
            Version of the dataset.
        """
        self.path = path
        self.zarr = zarr
        self.metadata = metadata
        self.version = version
        self.dataset = None
        self.dataset = open_dataset(self.path)

    def describe(self) -> None:
        """Print a description of the dataset."""
        print(f"📦 Path          : {self.path}")
        print(f"🔢 Format version: {self.version}")

    @property
    def name_to_index(self) -> Dict[str, int]:
        """Get a mapping of variable names to their indices."""
        return find(self.metadata, "name_to_index")

    @property
    def longitudes(self) -> NDArray[Any]:
        """Get the longitudes of the dataset."""
        try:
            return self.zarr.longitudes[:]
        except (KeyError, AttributeError):
            return self.zarr.longitude[:]

    @property
    def data(self) -> Any:
        """Get the data of the dataset."""
        try:
            return self.zarr.data
        except AttributeError:
            return self.zarr

    @property
    def first_date(self) -> datetime.datetime:
        """Get the first date of the dataset."""
        return datetime.datetime.fromisoformat(self.metadata["first_date"])

    @property
    def last_date(self) -> datetime.datetime:
        """Get the last date of the dataset."""
        return datetime.datetime.fromisoformat(self.metadata["last_date"])

    @property
    def frequency(self) -> str:
        """Get the frequency of the dataset."""
        return self.metadata["frequency"]

    @property
    def resolution(self) -> str:
        """Get the resolution of the dataset."""
        return self.metadata["resolution"]

    @property
    def field_shape(self) -> Optional[tuple]:
        """Get the field shape of the dataset."""
        return self.metadata.get("field_shape")

    @property
    def proj_string(self) -> Optional[str]:
        """Get the projection string of the dataset."""
        return self.metadata.get("proj_string")

    @property
    def shape(self) -> Optional[tuple]:
        """Get the shape of the dataset."""
        if self.data and hasattr(self.data, "shape"):
            return self.data.shape

    @property
    def n_missing_dates(self) -> Optional[int]:
        """Get the number of missing dates in the dataset."""
        if "missing_dates" in self.metadata:
            return len(self.metadata["missing_dates"])
        return None

    @property
    def uncompressed_data_size(self) -> Optional[int]:
        """Get the uncompressed data size of the dataset."""
        if self.data and hasattr(self.data, "dtype") and hasattr(self.data, "size"):
            return self.data.dtype.itemsize * self.data.size

    def info(self, detailed: bool, size: bool) -> None:
        """Print detailed information about the dataset.

        Parameters
        ----------
        detailed : bool
            Whether to print detailed information.
        size : bool
            Whether to print the size of the dataset.
        """
        print()
        print(f'📅 Start      : {self.first_date.strftime("%Y-%m-%d %H:%M")}')
        print(f'📅 End        : {self.last_date.strftime("%Y-%m-%d %H:%M")}')
        print(f"⏰ Frequency  : {self.frequency}")
        if self.n_missing_dates is not None:
            print(f"🚫 Missing    : {self.n_missing_dates:,}")
        print(f"🌎 Resolution : {self.resolution}")
        print(f"🌎 Field shape: {self.field_shape}")

        print()
        shape_str = "📐 Shape      : "
        if self.shape:
            shape_str += " × ".join(["{:,}".format(s) for s in self.shape])
        if self.uncompressed_data_size:
            shape_str += f" ({bytes(self.uncompressed_data_size)})"
        print(shape_str)
        self.print_sizes(size)
        print()
        rows = []

        if self.statistics_ready:
            stats = self.statistics
        else:
            stats = [["-"] * len(self.variables)] * 4

        for i, v in enumerate(self.variables):
            rows.append([i, v] + [x[i] for x in stats])

        print(
            table(
                rows,
                header=["Index", "Variable", "Min", "Max", "Mean", "Stdev"],
                align=[">", "<", ">", ">", ">", ">"],
                margin=3,
            )
        )

        if detailed:
            self.details()

        self.progress()
        if self.ready():
            self.probe()

        print()

    @property
    def variables(self) -> List[str]:
        """Get the list of variables in the dataset."""
        return [v[0] for v in sorted(self.name_to_index.items(), key=lambda x: x[1])]

    @property
    def total_size(self) -> Optional[int]:
        """Get the total size of the dataset."""
        return self.zarr.attrs.get("total_size")

    @property
    def total_number_of_files(self) -> Optional[int]:
        """Get the total number of files in the dataset."""
        return self.zarr.attrs.get("total_number_of_files")

    def print_sizes(self, size: bool) -> None:
        """Print the size and number of files in the dataset.

        Parameters
        ----------
        size : bool
            Whether to compute and print the size.
        """
        total_size = self.total_size
        n = self.total_number_of_files

        if total_size is None:
            if not size:
                return

            total_size, n = compute_directory_size(self.path)

        if total_size is not None:
            print(f"💽 Size       : {bytes(total_size)} ({bytes_to_human(total_size)})")
        if n is not None:
            print(f"📁 Files      : {n:,}")

    @property
    def statistics(self) -> tuple[list, list, list, list]:
        """Get the statistics of the dataset."""
        try:
            if self.dataset is not None:
                stats = self.dataset.statistics
                return stats["minimum"], stats["maximum"], stats["mean"], stats["stdev"]
        except AttributeError:
            return [["-"] * len(self.variables)] * 4

    @property
    def statistics_ready(self) -> bool:
        """Check if the statistics are ready."""
        for d in reversed(self.metadata.get("history", [])):
            if d["action"] == "compute_statistics_end":
                return True
        return False

    @property
    def statistics_started(self) -> Optional[datetime.datetime]:
        """Get the timestamp when statistics computation started."""
        for d in reversed(self.metadata.get("history", [])):
            if d["action"] == "compute_statistics_start":
                return datetime.datetime.fromisoformat(d["timestamp"])
        return None

    @property
    def build_flags(self) -> Optional[NDArray[Any]]:
        """Get the build flags of the dataset."""
        return self.zarr.get("_build_flags")

    @cached_property
    def copy_flags(self) -> Optional[NDArray[Any]]:
        """Get the copy flags of the dataset."""
        if "_copy" not in self.zarr:
            return None
        return self.zarr["_copy"][:]

    @property
    def copy_in_progress(self) -> bool:
        """Check if a copy operation is in progress."""
        if "_copy" not in self.zarr:
            return False

        start = self.zarr["_copy"].attrs.get("copy_start_timestamp")
        end = self.zarr["_copy"].attrs.get("copy_end_timestamp")
        if start and end:
            return False

        return not all(self.copy_flags)

    @property
    def build_lengths(self) -> Optional[NDArray]:
        """Get the build lengths of the dataset."""
        return self.zarr.get("_build_lengths")

    def progress(self) -> None:
        """Print the progress of dataset initialization or copying."""
        if self.copy_in_progress:
            copy_flags = self.copy_flags
            print("🪫  Dataset not ready, copy in progress.")
            assert isinstance(copy_flags, np.ndarray)
            total = len(copy_flags)
            built = copy_flags.sum()
            print(
                "📈 Progress:",
                progress(built, total, width=50),
                "{:.0f}%".format(built / total * 100),
            )
            return

        if self.build_flags is None:
            print("🪫 Dataset not initialised")
            return

        build_flags = self.build_flags

        build_lengths = self.build_lengths
        assert build_flags.size == build_lengths.size

        latest_write_timestamp = self.zarr.attrs.get("latest_write_timestamp")
        latest = datetime.datetime.fromisoformat(latest_write_timestamp) if latest_write_timestamp else None

        if not all(build_flags):
            if latest:
                print(f"🪫  Dataset not ready, last update {when(latest)}.")
            else:
                print("🪫  Dataset not ready.")
            total = sum(build_lengths)
            built = sum(ln if flag else 0 for ln, flag in zip(build_lengths, build_flags))
            print(
                "📈 Progress:",
                progress(built, total, width=50),
                "{:.0f}%".format(built / total * 100),
            )
            start = self.initialised
            if self.initialised:
                print(f"🕰️  Dataset initialised {when(start)}.")
                if built and latest:
                    speed = (latest - start) / built
                    eta = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + speed * (total - built)
                    print(f"🏁 ETA {when(eta)}.")
        else:
            if latest:
                print(f"🔋 Dataset ready, last update {when(latest)}.")
            else:
                print("🔋 Dataset ready.")
            if self.statistics_ready:
                print("📊 Statistics ready.")
            else:
                started = self.statistics_started
                if started:
                    print(f"⏳ Statistics not ready, started {when(started)}.")
                else:
                    print("⏳ Statistics not ready.")

    def brute_force_statistics(self) -> None:
        """Compute and print statistics for the dataset."""
        if self.dataset is None:
            return
        print("📊 Computing statistics...")
        # np.seterr(all="raise")

        nvars = self.dataset.shape[1]

        count = np.zeros(nvars, dtype=np.int64)
        sums = np.zeros(nvars, dtype=np.float32)
        squares = np.zeros(nvars, dtype=np.float32)

        minimum = np.full((nvars,), np.inf, dtype=np.float32)
        maximum = np.full((nvars,), -np.inf, dtype=np.float32)

        for i, chunk in enumerate(tqdm.tqdm(self.dataset, total=len(self.dataset))):
            values = chunk.reshape((nvars, -1))
            minimum = np.minimum(minimum, np.min(values, axis=1))
            maximum = np.maximum(maximum, np.max(values, axis=1))
            sums += np.sum(values, axis=1)
            squares += np.sum(values * values, axis=1)
            count += values.shape[1]

        mean = sums / count
        stats = [
            minimum,
            maximum,
            mean,
            np.sqrt(squares / count - mean * mean),
        ]

        rows = []

        for i, v in enumerate(self.variables):
            rows.append([i, v] + [x[i] for x in stats])

        print(
            table(
                rows,
                header=["Index", "Variable", "Min", "Max", "Mean", "Stdev"],
                align=[">", "<", ">", ">", ">", ">"],
                margin=3,
            )
        )


class NoVersion(Version):
    """Represents a dataset with no version."""

    @property
    def first_date(self) -> datetime.datetime:
        """Get the first date of the dataset."""
        monthly = find(self.metadata, "monthly")
        return datetime.datetime.fromisoformat(monthly["start"])

    @property
    def last_date(self) -> datetime.datetime:
        """Get the last date of the dataset."""
        monthly = find(self.metadata, "monthly")
        time = max([int(t) for t in find(self.metadata["earthkit-data"], "time")])
        assert isinstance(time, int), (time, type(time))
        if time > 100:
            time = time // 100
        return datetime.datetime.fromisoformat(monthly["stop"]) + datetime.timedelta(hours=time)

    @property
    def frequency(self) -> int:
        """Get the frequency of the dataset."""
        time = find(self.metadata["earthkit-data"], "time")
        return 24 // len(time)

    @property
    def statistics(self) -> tuple[list, list, list, list]:
        """Get the statistics of the dataset."""
        stats = find(self.metadata, "statistics_by_index")
        return stats["minimum"], stats["maximum"], stats["mean"], stats["stdev"]

    @property
    def statistics_ready(self) -> bool:
        """Check if the statistics are ready."""
        return find(self.metadata, "statistics_by_index") is not None

    @property
    def resolution(self) -> str:
        """Get the resolution of the dataset."""
        return find(self.metadata, "grid")

    def details(self) -> None:
        """Print details of the dataset."""
        pass

    def progress(self) -> None:
        """Print the progress of dataset initialization or copying."""
        pass

    def ready(self) -> bool:
        """Check if the dataset is ready.

        Returns
        -------
        bool
            True if the dataset is ready, False otherwise.
        """
        return True


class Version0_4(Version):
    """Represents version 0.4 of a dataset."""

    def details(self) -> None:
        """Print details of the dataset."""
        pass

    @property
    def initialised(self) -> datetime.datetime:
        """Get the initialization timestamp of the dataset."""
        return datetime.datetime.fromisoformat(self.metadata["creation_timestamp"])

    def statistics_ready(self) -> bool:
        """Check if the statistics are ready."""
        if not self.ready():
            return False
        build_flags = self.zarr["_build_flags"]
        return build_flags.attrs.get("_statistics_computed")

    def ready(self) -> bool:
        """Check if the dataset is ready."""
        if "_build_flags" not in self.zarr:
            return False

        build_flags = self.zarr["_build_flags"]
        if not build_flags.attrs.get("_initialised"):
            return False

        return all(build_flags)

    def _info(self, verbose: bool, history: bool, statistics: bool, **kwargs: Any) -> None:
        """Print information about the dataset.

        Parameters
        ----------
        verbose : bool
            Whether to print verbose information.
        history : bool
            Whether to print the history of the dataset.
        statistics : bool
            Whether to print statistics of the dataset.
        **kwargs : Any
            Additional keyword arguments.
        """
        z = self.zarr

        # for backward compatibility
        if "earthkit-data" in z.attrs:
            ekd_version = z.attrs["earthkit-data"].get("versions", {}).get("earthkit-data", "unkwown")
            print(f"earthkit-data version used to create this zarr: {ekd_version}. Not supported.")
            return

        version = z.attrs.get("version")
        versions = z.attrs.get("versions")
        if not versions:
            print(" Cannot find metadata information about versions.")
        else:
            print(f"Zarr format (version {version})", end="")
            print(f" created by earthkit-data={versions.pop('earthkit-data')}", end="")
            timestamp = z.attrs.get("creation_timestamp")
            timestamp = datetime.datetime.fromisoformat(timestamp)
            print(f" on {timestamp}", end="")
            versions = ", ".join([f"{k}={v}" for k, v in versions.items()])
            print(f" using {versions}", end="")
            print()


class Version0_6(Version):
    """Represents version 0.6 of a dataset."""

    @property
    def initialised(self) -> Optional[datetime.datetime]:
        """Get the initialization timestamp of the dataset."""
        for record in self.metadata.get("history", []):
            if record["action"] == "initialised":
                return datetime.datetime.fromisoformat(record["timestamp"])

        # Sometimes the first record is missing
        timestamps = sorted([datetime.datetime.fromisoformat(d["timestamp"]) for d in self.metadata.get("history", [])])
        if timestamps:
            return timestamps[0]

        return None

    def details(self) -> None:
        """Print details of the dataset."""
        print()
        for d in self.metadata.get("history", []):
            d = deepcopy(d)
            timestamp = d.pop("timestamp")
            timestamp = datetime.datetime.fromisoformat(timestamp)
            action = d.pop("action")
            versions = d.pop("versions")
            versions = ", ".join(f"{k}={v}" for k, v in versions.items())
            more = ", ".join(f"{k}={v}" for k, v in d.items())
            print(f"  {timestamp} : {action} ({versions}) {more}")
        print()

    def ready(self) -> bool:
        """Check if the dataset is ready."""
        if "_build_flags" not in self.zarr:
            return False

        build_flags = self.zarr["_build_flags"]
        return all(build_flags)

    @property
    def name_to_index(self) -> Dict[str, int]:
        """Get a mapping of variable names to their indices."""
        return {n: i for i, n in enumerate(self.metadata["variables"])}

    @property
    def variables(self) -> List[str]:
        """Get the list of variables in the dataset."""
        return self.metadata["variables"]

    @property
    def variables_metadata(self) -> dict:
        """Get the metadata for the variables."""
        return self.metadata.get("variables_metadata", {})


class Version0_12(Version0_6):
    """Represents version 0.12 of a dataset."""

    def details(self) -> None:
        """Print details of the dataset."""
        print()
        for d in self.metadata.get("history", []):
            d = deepcopy(d)
            timestamp = d.pop("timestamp")
            timestamp = datetime.datetime.fromisoformat(timestamp)
            action = d.pop("action")
            more = ", ".join(f"{k}={v}" for k, v in d.items())
            if more:
                more = f" ({more})"
            print(f"  {timestamp} : {action}{more}")
        print()

    @property
    def first_date(self) -> datetime.datetime:
        """Get the first date of the dataset."""
        return datetime.datetime.fromisoformat(self.metadata["start_date"])

    @property
    def last_date(self) -> datetime.datetime:
        """Get the last date of the dataset."""
        return datetime.datetime.fromisoformat(self.metadata["end_date"])


class Version0_13(Version0_12):
    """Represents version 0.13 of a dataset."""

    @property
    def build_flags(self) -> Optional[NDArray]:
        """Get the build flags for the dataset."""
        if "_build" not in self.zarr:
            return None
        build = self.zarr["_build"]
        return build.get("flags")

    @property
    def build_lengths(self) -> Optional[NDArray]:
        """Get the build lengths for the dataset."""
        if "_build" not in self.zarr:
            return None
        build = self.zarr["_build"]
        return build.get("lengths")


VERSIONS = {
    "0.0.0": NoVersion,
    "0.4.0": Version0_4,
    "0.6.0": Version0_6,
    "0.12.0": Version0_12,
    "0.13.0": Version0_13,
}


class InspectZarr(Command):
    """Command to inspect a zarr dataset."""

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        command_parser.add_argument("path", metavar="DATASET")
        command_parser.add_argument("--detailed", action="store_true")

        command_parser.add_argument("--progress", action="store_true")
        command_parser.add_argument("--statistics", action="store_true")
        command_parser.add_argument("--size", action="store_true", help="Print size")

    def run(self, args: Any) -> None:
        """Run the command.

        Parameters
        ----------
        args : Any
            The command arguments.
        """
        self.inspect_zarr(**vars(args))

    def inspect_zarr(
        self,
        path: str,
        progress: bool = False,
        statistics: bool = False,
        detailed: bool = False,
        size: bool = False,
        **kwargs: Any,
    ) -> None:
        """Inspect a zarr dataset.

        Parameters
        ----------
        path : str
            Path to the dataset.
        progress : bool, optional
            Whether to print progress, by default False.
        statistics : bool, optional
            Whether to compute and print statistics, by default False.
        detailed : bool, optional
            Whether to print detailed information, by default False.
        size : bool, optional
            Whether to print the size of the dataset, by default False.
        **kwargs : Any
            Additional keyword arguments.
        """
        version = self._info(path)

        dotted_line()
        version.describe()

        try:
            if progress:
                return version.progress()

            if statistics:
                return version.brute_force_statistics()

            version.info(detailed, size)

        except Exception as e:
            LOG.error("Error inspecting zarr file '%s': %s", path, e)

            print(type(version))
            raise

    def _info(self, path: str) -> Version:
        """Get version information of the dataset.

        Parameters
        ----------
        path : str
            Path to the dataset.

        Returns
        -------
        Version
            The version object of the dataset.
        """
        z = open_zarr(zarr_lookup(path))

        metadata = dict(z.attrs)
        version = metadata.get("version", "0.0.0")
        if isinstance(version, int):
            version = f"0.{version}"

        version = semantic_version.Version.coerce(version)

        versions = {semantic_version.Version.coerce(k): v for k, v in VERSIONS.items()}

        candidate = None
        for v, klass in sorted(versions.items()):
            if version >= v:
                candidate = klass

        return candidate(path, z, metadata, version)


command = InspectZarr
