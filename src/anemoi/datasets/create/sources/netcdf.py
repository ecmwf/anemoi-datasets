# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from . import source_registry
from .xarray import XarraySourceBase


@source_registry.register("netcdf")
class NetCDFSource(XarraySourceBase):
    """Read fields from NetCDF files (one ``path``, or a ``{date}`` pattern).

    Supports both the analysis layout and the trajectory layout (where each
    file is one forecast run, located from its basetime); see
    :class:`XarraySourceBase`.
    """

    emoji = "📁"
