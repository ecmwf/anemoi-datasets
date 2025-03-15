# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from typing import Any
from typing import Dict

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list

from .legacy import legacy_filter


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, orog: str, z: str = "z") -> ekd.FieldList:
    """Convert orography [m] to z (geopotential height).

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : FieldList
        List of input fields.
    orog : str
        Orography parameter.
    z : str, optional
        Geopotential height parameter. Defaults to "z".

    Returns
    -------
    FieldList
        List of fields with geopotential height.
    """
    result = []
    processed_fields: Dict[tuple, Dict[str, Any]] = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        if param == orog:
            key = tuple(key.items())

            if param in processed_fields[key]:
                raise ValueError(f"Duplicate field {param} for {key}")

            output = f.to_numpy(flatten=True) * 9.80665
            result.append(new_field_from_numpy(f, output, param=z))
        else:
            result.append(f)

    return new_fieldlist_from_list(result)
