# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from typing import Any

from anemoi.utils.nexus import add_nexus_record_arguments
from anemoi.utils.nexus import nexus_record
from anemoi.utils.nexus import record_attributes
from anemoi.utils.nexus import write_nexus_record

from . import Command

LOG = logging.getLogger(__name__)

#: The statistics arrays of a dataset, copied into ``metadata.statistics``.
STATISTICS = ("mean", "stdev", "minimum", "maximum")


def dataset_nexus_record(path: str, attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Nexus record of the zarr dataset at *path*: its name (the
    directory name without ``.zarr``), its ``uuid``, and as ``metadata`` its
    zarr attributes with the statistics and the ``data`` array's shape, dtype
    and chunks added; then the *attributes* (owner, projects, licenses, ...).

    Parameters
    ----------
    path : str
        The path of the dataset (a ``.zarr`` directory).
    attributes : dict, optional
        The record attributes (see :mod:`anemoi.utils.nexus`).

    Returns
    -------
    dict
        The record, for ``nexus-client create datasets NAME --file``.
    """
    import zarr

    path = os.path.abspath(path.rstrip("/"))
    if not path.endswith(".zarr"):
        raise ValueError(f"{path}: a dataset path must end in .zarr")
    z = zarr.open(path, mode="r")
    metadata = dict(z.attrs.asdict())
    uuid = metadata.get("uuid")
    if not uuid:
        raise ValueError(f"{path}: the dataset has no 'uuid' attribute")
    statistics = {key: z[key][:].tolist() for key in STATISTICS if key in z}
    if statistics:
        metadata["statistics"] = statistics
    if "data" in z:
        data = z["data"]
        metadata["shape"] = list(data.shape)
        metadata["dtype"] = str(data.dtype)
        metadata["chunks"] = list(data.chunks)
    name = os.path.basename(path)[: -len(".zarr")]
    return nexus_record(uuid=uuid, name=name, metadata=metadata, attributes=attributes)


class NexusRecord(Command):
    """Print a dataset's record for Anemoi Nexus (``nexus-client create datasets NAME --file``)."""

    timestamp = False

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        command_parser.add_argument("path", metavar="DATASET", help="Path of the dataset (a .zarr directory).")
        add_nexus_record_arguments(command_parser)

    def run(self, args: Any) -> None:
        """Print or write the record.

        Parameters
        ----------
        args : Any
            The command arguments.
        """
        write_nexus_record(dataset_nexus_record(args.path, record_attributes(args)), args.output)


command = NexusRecord
