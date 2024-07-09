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

from .fieldlist import XarrayFieldList

LOG = logging.getLogger(__name__)


def execute(context, dates, dataset, options, flavour=None, *args, **kwargs):
    import xarray as xr

    context.trace("🌐", dataset, options)

    if isinstance(dataset, str) and ".zarr" in dataset:
        data = xr.open_zarr(dataset, **options)
    else:
        data = xr.open_dataset(dataset, **options)

    fs = XarrayFieldList.from_xarray(data, flavour)
    return MultiFieldList([fs.sel(valid_datetime=date, **kwargs) for date in dates])
