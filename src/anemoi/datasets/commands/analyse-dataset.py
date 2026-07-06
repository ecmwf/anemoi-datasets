# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import os
import sys
from typing import Any

from . import Command

LOG = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class AnalyseDataset(Command):
    """Analyse a dataset and output a JSON catalogue record.

    The JSON document contains the top-level keys 'type' (always
    "dataset"), 'name' and 'record'. The 'record' is the catalogue
    record that can be registered with
    ``anemoi-registry register <file.json>``.
    """

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        command_parser.add_argument("path", metavar="DATASET", help="Path (or name) of a zarr dataset.")
        command_parser.add_argument(
            "-o",
            "--output",
            metavar="FILE",
            help="Write the JSON to FILE instead of the standard output.",
        )

    def run(self, args: Any) -> None:
        """Run the command.

        Parameters
        ----------
        args : Any
            The command arguments.
        """
        from anemoi.datasets import __version__
        from anemoi.datasets import open_dataset
        from anemoi.datasets.usage.store import open_zarr_store

        z, path = open_zarr_store(args.path, return_path=True)

        if not path.startswith("s3://") and not path.startswith("http://") and not path.startswith("https://"):
            path = os.path.abspath(path)

        name, _ = os.path.splitext(os.path.basename(path))

        ds = open_dataset(path)

        metadata = dict(z.attrs)

        try:
            metadata["statistics"] = {k: v.tolist() for k, v in ds.statistics.items()}
        except (AttributeError, KeyError):
            if "statistics" in metadata:
                LOG.warning("Found statistics in metadata, but not in dataset.")
            else:
                LOG.warning("No statistics found in metadata.")
                metadata["statistics"] = dict(mean=[], stdev=[], minimum=[], maximum=[])

        shape = z["data"].shape
        if "shape" in metadata:
            assert tuple(metadata["shape"]) == shape, (metadata["shape"], shape)
        metadata["shape"] = list(shape)

        if "dtype" in metadata:
            assert metadata["dtype"] == str(ds.dtype), (metadata["dtype"], ds.dtype)
        metadata["dtype"] = str(ds.dtype)

        if "chunks" in metadata:
            assert tuple(metadata["chunks"]) == tuple(ds.chunks), (metadata["chunks"], ds.chunks)
        metadata["chunks"] = list(ds.chunks)

        result = {
            "type": "dataset",
            "name": name,
            "version": "1.0",
            "anemoi_datasets_version": __version__,
            "path": path,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "record": {
                "name": name,
                "metadata": metadata,
            },
        }

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=_json_default)
                f.write("\n")
            LOG.info(f"Written {args.output}")
        else:
            json.dump(result, sys.stdout, indent=2, default=_json_default)
            sys.stdout.write("\n")


command = AnalyseDataset
