# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import earthkit.data as ekd
import planetary_computer
import pystac
import pystac_client

from anemoi.datasets.create.types import DateList

from . import source_registry
from .xarray import XarraySourceBase
from .xarray import load_one

_MEDIA_TYPE_XR_ENGINES = {
    "application/vnd+zarr": "zarr",
    "application/netcdf": "h5netcdf",
}


@source_registry.register("planetary_computer")
class PlanetaryComputerSource(XarraySourceBase):
    """An Xarray data source for the planetary_computer.

    This source is intended to handle two types of collections:
    - data are consolidated in one Zarr store under a single collection-level asset with
    the key `zarr-abfs`.
    - data are split across multiple items and assets thereof.

    For multipart collections, pass ``search_params`` with ``datetime``,
    ``filter`` (CQL2 JSON or text), and ``variable_key_map`` to control STAC
    item search and asset key resolution.

    """

    emoji = "🪐"

    def __init__(
        self,
        context: Any,
        data_catalog_id: str,
        version: str = "v1",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise the PlanetaryComputerSource.

        Parameters
        ----------
        context : Any
            Context for the data source.
        data_catalog_id : str
            ID of the STAC collection to query.
        version : str
            Version of the Planetary Computer STAC API to use (default "v1").
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments. For multipart collections, accepts
            ``search_params`` dict with keys ``datetime``, ``filter`` (CQL2),
            and ``variable_key_map``.

        """
        self.flavour = kwargs.pop("flavour", None)
        self.patch = kwargs.pop("patch", None)

        loglevel = kwargs.pop("azure_log_level", "WARNING")
        if isinstance(loglevel, (str, int)):
            logging.getLogger("azure").setLevel(loglevel)

        catalog = pystac_client.Client.open(
            f"https://planetarycomputer.microsoft.com/api/stac/{version}/",
            modifier=planetary_computer.sign_inplace,
        )
        collection = catalog.get_collection(data_catalog_id)

        # singlepart collections have a collection-level asset with key "zarr-abfs"
        if asset := collection.assets.get("zarr-abfs"):
            self.options = {
                **kwargs.pop("options", {}),
                "storage_options": asset.extra_fields.get("xarray:storage_options"),
                **asset.extra_fields["xarray:open_kwargs"],
            }
            super().__init__(context, url=asset.href, *args, **kwargs)
        else:
            search_params = kwargs.pop("search_params", {})
            self.uris = self._do_multipart_query(
                catalog,
                data_catalog_id,
                kwargs.get("param"),
                search_params,
            )
            super().__init__(context, **kwargs)
        self._singlepart = asset is not None

    def _do_multipart_query(
        self,
        catalog: pystac_client.Client,
        collection_id: str,
        param: str | list[str] | None,
        search_params: dict,
    ) -> dict[str, tuple[dict, datetime | None]]:
        """Query STAC and resolve assets for multipart collections.

        Parameters
        ----------
        catalog : pystac_client.Client
            STAC catalogue client to use in querying.
        collection_id : str
            ID of the STAC collection to query.
        param : str | list[str] | None
            Asset key(s) to load from each item. If None, all are loaded.
        search_params : dict
            STAC search parameters:
            - `datetime`: passed to `pystac_client.Client.search(datetime=...)`.
            - `variable_key_map`: mapping of data variable names to STAC asset keys
              for collections where they differ.
            - `filter`: CQL2 filter (dict for cql2-json, str for cql2-text) passed
              directly to `pystac_client.Client.search(filter=...)`.

        Returns
        -------
        dict[str, tuple[dict, datetime | None]]
            Dictionary mapping resolved asset ABFS URIs to a tuple of Xarray open kwargs
            and the asset's item datetime. Datetime is typically None for items
            representing a time range rather than a single timestamp.

        """
        # with some collections, data variable name != STAC key, must resolve
        key_var_map = search_params.get("variable_key_map", {})
        params = [param] if isinstance(param, str) else (param or [])
        stac_keys = [key_var_map.get(p, p) for p in params]

        search = catalog.search(
            collections=[collection_id],
            datetime=search_params.get("datetime"),
            filter=search_params.get("filter"),
        )

        # assets.* not queryable server-side, filter client-side
        uris: dict[str, tuple[dict, datetime | None]] = {}
        for item in search.items():
            for sk in stac_keys or item.assets:
                if sk in item.assets:
                    # kwargs per item to enable different locations and / or options
                    abfs_uri, open_kwargs = _resolve_asset(item.assets[sk])
                    # item.datetime set for single-timestamp items
                    # None for range items that set start_datetime and end_datetime
                    uris[abfs_uri] = (open_kwargs, item.datetime)

        if not uris:
            msg = f"No STAC items or assets found: collection={collection_id}, " f"search_params={search_params}"
            raise ValueError(msg)
        return uris

    def execute(self, dates: DateList) -> ekd.FieldList:
        """Execute data loading for given dates.

        Parameters
        ----------
        dates : DateList
            List of dates for which to load data.

        Returns
        -------
        ekd.FieldList
            Loaded data field(s).

        """
        if self._singlepart:
            return super().execute(dates)
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
                    flavour=self.flavour,  # type: ignore[arg-type]
                    patch=self.patch,
                    **self.kwargs,
                ),
            )
        return ekd.concat(*result)


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

    open_kwargs.setdefault("decode_timedelta", True)
    return abfs_uri, open_kwargs
