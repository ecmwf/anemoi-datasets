# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json as json_mod
import logging
from pathlib import Path
from typing import IO
from typing import Any
from typing import cast

from . import source_registry
from .xarray import XarraySourceBase

LOG = logging.getLogger(__name__)


@source_registry.register("xarray_kerchunk")
class XarrayKerchunkSource(XarraySourceBase):
    """An Xarray data source that uses the `kerchunk` engine."""

    emoji = "🧱"

    def __init__(
        self,
        context: Any,
        json: str | dict,
        *args: str,
        json_options: dict[str, Any] | None = None,
        storage_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the XarrayKerchunkSource.

        Parameters
        ----------
        context : Any
            Context for the data source.
        json : str | dict
            Kerchunk references as one of: a JSON string, a local JSON file path, a remote JSON file path, or a
            dictionary of Kerchunk references.
        *args
            Additional positional arguments.
        json_options : dict[str, Any] | None, default = None
            Storage options passed to `fsspec.open` for opening a remote references JSON file.
        storage_options : dict[str, Any] | None, default = None
            Storage options passed to `xarray.open_zarr` for loading remote data from references.
        **kwargs
            Additional keyword arguments.

        """
        super().__init__(context, *args, **kwargs)

        self.path_or_url = "reference://"

        if isinstance(json, str):
            if json.lstrip().startswith("{"):  # "looks like" JSON-formatted string of references
                LOG.info("Using references JSON string.")
                json = json_mod.loads(json)

            elif (local_json := Path(json).resolve()).exists():
                LOG.info("Using local references JSON file: %s", local_json)
                with local_json.open() as f:
                    json = json_mod.load(f)

            else:
                import fsspec  # type: ignore[import-untyped]

                LOG.info("Using remote references JSON file: %s", json)
                options = dict(json_options or {})
                options["urlpath"] = json  # ensure urlpath arg not pre-set in json_options
                with fsspec.open(**options) as f:
                    json = json_mod.load(cast("IO[bytes]", f))

        elif isinstance(json, dict):
            LOG.info("Using references JSON dictionary.")

        else:
            msg = f"'json' must be a JSON string, file path, or dictionary, got {type(json)}."
            raise TypeError(msg)

        self.options = {
            "engine": "zarr",
            "backend_kwargs": {
                "consolidated": False,
                "storage_options": {
                    **(storage_options or {}),
                    "fo": json,  # last to override any setting in storage_options
                },
            },
        }
