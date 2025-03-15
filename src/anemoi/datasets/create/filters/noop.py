# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import earthkit.data as ekd

from .legacy import legacy_filter


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, *args: Any, **kwargs: Any) -> ekd.FieldList:
    """No operation filter that returns the input as is.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : ekd.FieldList
        List of input fields.
    *args : Any
        Additional arguments.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    List[Any]
        The input list of fields.
    """
    return input
