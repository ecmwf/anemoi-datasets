# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import planetary_computer
import pystac
import pystac_client
from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.source import Source
from anemoi.datasets.create.types import DateList

from . import source_registry
from .xarray import XarraySourceBase
from .xarray_support import load_one

if TYPE_CHECKING:
    from datetime import datetime

_MEDIA_TYPE_XR_ENGINES = {
    "application/vnd+zarr": "zarr",
    "application/netcdf": "h5netcdf",
}


@source_registry.register("planetary_computer")
class PlanetaryComputerSource(XarraySourceBase):
    """An Xarray data source for the planetary_computer."""

    emoji = "🪐"

    def __init__(self, context, data_catalog_id, version="v1", *args, **kwargs: dict):
        self.data_catalog_id = data_catalog_id
        self.flavour = kwargs.pop("flavour", None)
        self.patch = kwargs.pop("patch", None)
        self.options = kwargs.pop("options", {})

        catalog = pystac_client.Client.open(
            f"https://planetarycomputer.microsoft.com/api/stac/{version}/",
            modifier=planetary_computer.sign_inplace,
        )
        collection = catalog.get_collection(self.data_catalog_id)

        asset = collection.assets["zarr-abfs"]

        if "xarray:storage_options" in asset.extra_fields:
            self.options["storage_options"] = asset.extra_fields["xarray:storage_options"]

        self.options.update(asset.extra_fields["xarray:open_kwargs"])

        super().__init__(context, url=asset.href, *args, **kwargs)


def _signed_url_to_abfs(url: str) -> tuple[str, dict]:
    """Convert a signed Azure Blob HTTPS URL to an abfs:// URI and storage options.

    Parameters
    ----------
    url : str
        Signed Azure Blob HTTPS URL to convert.

    Returns
    -------
    tuple[str, dict]
        Tuple of the resolved abfs:// URI and storage options dictionary.

    """
    parsed = urlparse(url)
    if parsed.hostname:
        account_name = parsed.hostname.split(".")[0]
    else:
        msg = f"Could not identify hostname from URL: {url}"
        raise ValueError(msg)
    _, container, blob_path = parsed.path.split("/", 2)
    return (
        f"abfs://{container}/{blob_path}",
        {"account_name": account_name, "sas_token": parsed.query},
    )


def _infer_engine(media_type: str | None, uri: str) -> str | None:
    """Infer the Xarray engine from media type, falling back to file extension.

    Parameters
    ----------
    media_type : str | None
        Media type (MIME) of the asset, if provided.
    uri : str
        URI of the asset.

    Returns
    -------
    engine : str | None
        Inferred Xarray engine, None if it cannot be determined.

    """
    engine = _MEDIA_TYPE_XR_ENGINES.get(media_type or "")
    if engine is None:
        if uri.endswith(".nc"):
            engine = "h5netcdf"
        elif ".zarr" in uri:
            engine = "zarr"
    return engine


def _resolve_asset(asset: pystac.Asset) -> tuple[str, dict]:
    """Resolve a STAC asset to an abfs:// URI and storage options.

    Parameters
    ----------
    asset : pystac.Asset
        STAC asset to resolve.

    Returns
    -------
    tuple[str, dict]
        Tuple of the resolved abfs:// URI and storage options dictionary.

    """
    href = asset.href
    if href.startswith("https://"):
        abfs_uri, storage_options = _signed_url_to_abfs(href)
    elif href.startswith("abfs://"):
        abfs_uri = href
        storage_options = asset.extra_fields.get("xarray:storage_options", {})
    else:
        msg = f"Unsupported asset href scheme, expected HTTPS URL or ABFS URI: {href}"
        raise ValueError(msg)

    open_kwargs = dict(asset.extra_fields.get("xarray:open_kwargs", {}))

    # storage_options nested inside open_kwargs for some collections
    if "storage_options" in open_kwargs:
        storage_options = {**open_kwargs.pop("storage_options"), **storage_options}
    open_kwargs["storage_options"] = storage_options

    if "engine" not in open_kwargs:
        engine = _infer_engine(asset.media_type, abfs_uri)
        if engine:
            open_kwargs["engine"] = engine

    return abfs_uri, open_kwargs


@source_registry.register("planetary_computer_multipart")
class MultipartPlanetaryComputerSource(Source):
    """An Xarray data source for the planetary_computer_multipart.

    This source is intended to handle collections where data is split across multiple
    items and assets thereof, as opposed to a single collection-level asset under the
    `zarr-abfs` key.

    """

    emoji = "🪐"

    def __init__(
        self,
        context: Any,
        collection_id: str,
        query: dict,
        version: str = "v1",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise the MultipartPlanetaryComputerSource.

        Parameters
        ----------
        context : Any
            Context for the data source.
        collection_id : str
            ID of the STAC collection to query.
        query : dict
            Dictionary of query parameters to filter STAC items. Recommended to include
            a datetime filter under `query.datetime` to reduce query time and the number
            of results to filter. See `pystac_client.Client.search` for accepted
            formats.
        version : str
            Version of the Planetary Computer STAC API to use (default "v1").
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        """
        super().__init__(context, *args, **kwargs)

        self.flavour = kwargs.pop("flavour", None)
        self.patch = kwargs.pop("patch", None)
        loglevel = kwargs.pop("azure_log_level", "WARNING")
        param = kwargs.pop("param", None)
        self.kwargs = kwargs

        if isinstance(loglevel, (str, int)):
            logging.getLogger("azure").setLevel(loglevel)

        dt = query.pop("datetime", None)
        # build CQL2 AND filter from remaining query fields, if any
        filter_args = [
            {"op": "=", "args": [{"property": k}, v]}
            for k, v in query.items()
        ]
        cql2_filter = None
        if len(filter_args) > 1:
            cql2_filter = {"op": "and", "args": filter_args}
        elif len(filter_args) == 1:
            cql2_filter = filter_args[0]

        client = pystac_client.Client.open(
            f"https://planetarycomputer.microsoft.com/api/stac/{version}/",
            modifier=planetary_computer.sign_inplace,
        )
        search = client.search(
            collections=[collection_id],
            datetime=dt,
            filter=cql2_filter,
        )

        params = [param] if isinstance(param, str) else (param or [])

        # assets.* not queryable server-side, filter client-side
        self.uris: dict[str, tuple[dict, datetime | None]] = {}
        for item in search.items():
            for p in (params or item.assets):
                if p in item.assets:
                    # kwargs per item to enable different locations and / or options
                    abfs_uri, open_kwargs = _resolve_asset(item.assets[p])
                    # item.datetime set for single-timestamp items
                    # None for range items that set start_datetime and end_datetime
                    self.uris[abfs_uri] = (open_kwargs, item.datetime)

        if not self.uris:
            msg = (
                f"No STAC items or assets found: collection={collection_id}, "
                f"query={query}"
            )
            raise ValueError(msg)

    def execute(self, dates: DateList) -> MultiFieldList:
        """Execute data loading for given dates.

        Parameters
        ----------
        dates : DateList
            List of dates for which to load data.

        Returns
        -------
        earthkit.data.MultiFieldList
            Loaded data fields.

        """
        result = []
        for abfs_path, (xr_kwargs, item_dt) in self.uris.items():
            if item_dt is not None:
                # for single-timestamp items, only pass dates that match the timestamp
                naive_dt = item_dt.replace(tzinfo=None) if item_dt.tzinfo else item_dt
                matching = [d for d in dates if d == naive_dt]
                if not matching:
                    continue
                load_dates = matching
            else:
                # for timestamp range items, pass all dates and let load_one select
                load_dates = dates
            result.append(
                load_one(
                    self.emoji,
                    self.context,
                    [d.isoformat() for d in load_dates],
                    abfs_path,
                    options=xr_kwargs,
                    flavour=self.flavour,
                    patch=self.patch,
                    **self.kwargs,
                ),
            )
        return MultiFieldList(result)
