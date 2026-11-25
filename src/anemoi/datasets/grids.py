# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.transform.grids import LatLonGrid


def check_grids_compatible(
    reference: "LatLonGrid",
    other: "LatLonGrid",
    where: str,
    reorder_hint: str = "",
) -> None:
    """Raise unless `other` has the same grid points in the same order as `reference`."""
    comparison = reference.compare(other)
    if comparison.is_identical:
        return
    if comparison.is_reorderable:
        raise ValueError(
            f"{where}: same grid points but different point ordering "
            f"(e.g. reversed latitude or 0-360 vs -180-180 longitude "
            f"convention).{reorder_hint}"
        )
    raise comparison.error or ValueError(f"{where}: incompatible grid")
