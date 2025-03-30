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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import earthkit.data as ekd
import xarray as xr
from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.sources.patterns import iterate_patterns
from anemoi.datasets.data.stores import name_to_zarr_store

from ..legacy import legacy_source
from .fieldlist import XarrayFieldList

LOG = logging.getLogger(__name__)


def check(what: str, ds: xr.Dataset, paths: List[str], **kwargs: Any) -> None:
    """Checks if the dataset has the expected number of fields.

    Parameters
    ----------
    what : str
        Description of what is being checked.
    ds : xr.Dataset
        The dataset to check.
    paths : List[str]
        List of paths.
    **kwargs : Any
        Additional keyword arguments.
    """
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, {what}s={paths})")


def load_one(
    emoji: str,
    context: Any,
    dates: List[str],
    dataset: Union[str, xr.Dataset],
    *,
    options: Optional[Dict[str, Any]] = None,
    flavour: Optional[str] = None,
    patch: Optional[Any] = None,
    **kwargs: Any,
) -> ekd.FieldList:
    """Loads a single dataset.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    dates : List[str]
        List of dates.
    dataset : Union[str, xr.Dataset]
        The dataset to load.
    options : Dict[str, Any], optional
        Additional options for loading the dataset.
    flavour : Optional[str], optional
        Flavour of the dataset.
    patch : Optional[Any], optional
        Patch for the dataset.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        The loaded dataset.
    """

    """
    We manage the S3 client ourselves, bypassing fsspec and s3fs layers, because sometimes something on the stack
    zarr/fsspec/s3fs/boto3 (?) seem to flags files as missing when they actually are not (maybe when S3 reports some sort of
    connection error). In that case,  Zarr will silently fill the chunks that could not be downloaded with NaNs.
    See https://github.com/pydata/xarray/issues/8842

    We have seen this bug triggered when we run many clients in parallel, for example, when we create a new dataset using `xarray-zarr`.
    """

    if options is None:
        options = {}

    context.trace(emoji, dataset, options, kwargs)

    if isinstance(dataset, str) and ".zarr" in dataset:
        data = xr.open_zarr(name_to_zarr_store(dataset), **options)
    elif "planetarycomputer" in dataset:
        store = name_to_zarr_store(dataset)
        if "store" in store:
            data = xr.open_zarr(**store)
        if "filename_or_obj" in store:
            data = xr.open_dataset(**store)
    else:
        data = xr.open_dataset(dataset, **options)

    fs = XarrayFieldList.from_xarray(data, flavour=flavour, patch=patch)

    if len(dates) == 0:
        result = fs.sel(**kwargs)
    else:
        print("dates", dates, kwargs)
        result = MultiFieldList([fs.sel(valid_datetime=date, **kwargs) for date in dates])

    if len(result) == 0:
        LOG.warning(f"No data found for {dataset} and dates {dates} and {kwargs}")
        LOG.warning(f"Options: {options}")

        for i, k in enumerate(fs):
            a = ["valid_datetime", k.metadata("valid_datetime", default=None)]
            for n in kwargs.keys():
                a.extend([n, k.metadata(n, default=None)])
            print([str(x) for x in a])

            if i > 16:
                break

        # LOG.warning(data)

    return result


def load_many(emoji: str, context: Any, dates: List[datetime.datetime], pattern: str, **kwargs: Any) -> ekd.FieldList:
    """Loads multiple datasets.

    Parameters
    ----------
    emoji : str
        Emoji for tracing.
    context : Any
        Context object.
    dates : List[str]
        List of dates.
    pattern : str
        Pattern for loading datasets.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        The loaded datasets.
    """
    result = []

    for path, dates in iterate_patterns(pattern, dates, **kwargs):
        result.append(load_one(emoji, context, dates, path, **kwargs))

    return MultiFieldList(result)


@legacy_source("xarray")
def execute(context: Any, dates: List[str], url: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
    """Executes the loading of datasets.

    Parameters
    ----------
    context : Any
        Context object.
    dates : List[str]
        List of dates.
    url : str
        URL pattern for loading datasets.
    *args : Any
        Additional arguments.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        The loaded datasets.
    """
    return load_many("🌐", context, dates, url, *args, **kwargs)
