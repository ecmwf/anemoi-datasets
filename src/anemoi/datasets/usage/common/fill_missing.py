# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Layout-agnostic ``fill_missing_dates`` factory.

The interpolation kernels in :mod:`anemoi.datasets.usage.gridded.fill_missing`
read flanking samples along axis 0 and broadcast across the remaining axes,
so they apply unchanged to a 5-D trajectory array.  Re-exporting from
``common`` lets the trajectory factory loader find the symbol.
"""

from anemoi.datasets.usage.gridded.fill_missing import fill_missing_dates_factory  # noqa: F401
