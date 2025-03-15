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
from typing import Hashable
from typing import List
from typing import Tuple

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list

from .legacy import legacy_filter


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, params: List[str], output: str) -> ekd.FieldList:
    """Computes the sum over a set of variables.

    Args:
        context (Any): The execution context.
        input (List[Any]): The list of input fields.
        params (List[str]): The list of parameters to sum over.
        output (str): The name for the output field.

    Returns:
        ekd.FieldList: The resulting FieldArray with summed fields.
    """
    result = []

    needed_fields: Dict[Tuple[Hashable, ...], Dict[str, ekd.Field]] = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        if param in params:
            key = tuple(key.items())

            if param in needed_fields[key]:
                raise ValueError(f"Duplicate field {param} for {key}")

            needed_fields[key][param] = f
        else:
            result.append(f)

    for keys, values in needed_fields.items():

        if len(values) != len(params):
            raise ValueError("Missing fields")

        s = None
        for k, v in values.items():
            c = v.to_numpy(flatten=True)
            if s is None:
                s = c
            else:
                s += c
        result.append(new_field_from_numpy(values[list(values.keys())[0]], s, param=output))

    return new_fieldlist_from_list(result)
