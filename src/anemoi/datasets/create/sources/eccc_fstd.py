# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from . import source_registry
from .xarray import XarraySourceBase


@source_registry.register("eccc_fstd")
class XarrayECCCSource(XarraySourceBase):
    """An Xarray data source that uses the `fstd` engine."""

    emoji = "🍁"
    options = {"engine": "fstd"}
