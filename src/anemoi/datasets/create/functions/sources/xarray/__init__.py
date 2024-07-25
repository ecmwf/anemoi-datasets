# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.data.stores import name_to_zarr_store

from .. import iterate_patterns
from .fieldlist import XarrayFieldList

LOG = logging.getLogger(__name__)


def check(what, ds, paths, **kwargs):
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, {what}s={paths})")


def load_one(emoji, context, dates, dataset, options={}, flavour=None, **kwargs):
    import xarray as xr

    """
    We manage the S3 client ourselve, bypassing fsspec and s3fs layers, because sometimes something on the stack
    zarr/fsspec/s3fs/boto3 (?) seem to flags files as missing when they actually are not (maybe when S3 reports some sort of
    connection error). In that case,  Zarr will silently fill the chunks that could not be downloaded with NaNs.
    See https://github.com/pydata/xarray/issues/8842

    We have seen this bug triggered when we run many clients in parallel, for example, when we create a new dataset using `xarray-zarr`.
    """

    context.trace(emoji, dataset, options)

    if isinstance(dataset, str) and ".zarr" in dataset:
        data = xr.open_zarr(name_to_zarr_store(dataset), **options)
    else:
        data = xr.open_dataset(dataset, **options)

    fs = XarrayFieldList.from_xarray(data, flavour)
    result = MultiFieldList([fs.sel(valid_datetime=date, **kwargs) for date in dates])

    if len(result) == 0:
        LOG.warning(f"No data found for {dataset} and dates {dates}")
        LOG.warning(f"Options: {options}")
        LOG.warning(data)

    return result


def load_many(emoji, context, dates, pattern, **kwargs):

    result = []

    for path, dates in iterate_patterns(pattern, dates, **kwargs):
        result.append(load_one(emoji, context, dates, path, **kwargs))

    return MultiFieldList(result)


def execute(context, dates, url, *args, **kwargs):
    return load_many("🌐", context, dates, url, *args, **kwargs)
