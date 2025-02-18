# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Dict
from typing import List

LOG = logging.getLogger(__name__)


def patch_attributes(ds: Any, attributes: Dict[str, Dict[str, Any]]) -> Any:
    for name, value in attributes.items():
        variable = ds[name]
        variable.attrs.update(value)

    return ds


def patch_coordinates(ds: Any, coordinates: List[str]) -> Any:
    for name in coordinates:
        ds = ds.assign_coords({name: ds[name]})

    return ds


PATCHES = {
    "attributes": patch_attributes,
    "coordinates": patch_coordinates,
}


def patch_dataset(ds: Any, patch: Dict[str, Dict[str, Any]]) -> Any:
    for what, values in patch.items():
        if what not in PATCHES:
            raise ValueError(f"Unknown patch type {what!r}")

        ds = PATCHES[what](ds, values)

    return ds
