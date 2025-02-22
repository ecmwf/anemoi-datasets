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


def execute(context: Any, input: ekd.FieldList, **kwargs: Any) -> Any:
    """
    Create a pipeline that returns an empty result.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : List[Any]
        List of input fields.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        An empty result.
    """
    return ekd.from_source("empty")
