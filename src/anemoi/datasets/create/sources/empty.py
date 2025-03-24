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

import earthkit.data as ekd

from .legacy import legacy_source


@legacy_source(__file__)
def execute(context: Any, dates: List[str], **kwargs: Any) -> ekd.FieldList:
    """Executes the loading of an empty data source.

    Parameters
    ----------
    context : object
        The context in which the function is executed.
    dates : list
        List of dates for which data is to be loaded.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        Loaded empty data source.
    """
    return ekd.from_source("empty")
