# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import List


def execute(context: Any, input: List[Any], *args: Any, **kwargs: Any) -> List[Any]:
    """No operation filter that returns the input as is.

    Args:
        context (Any): The context in which the function is executed.
        input (List[Any]): List of input fields.
        *args (Any): Additional arguments.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        List[Any]: The input list of fields.
    """
    return input
