# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from earthkit.data import FieldList
from earthkit.data.core.fieldlist import MultiFieldList

LOG = logging.getLogger(__name__)


def _flatten(ds: MultiFieldList | FieldList) -> list:
    """Flattens a MultiFieldList or FieldList into a list of FieldList objects.

    Parameters
    ----------
    ds : Union[MultiFieldList, FieldList]
        The dataset to flatten.

    Returns
    -------
    list
        A list of FieldList objects.
    """
    if isinstance(ds, MultiFieldList):
        return [_tidy(f) for s in ds._indexes for f in _flatten(s)]
    return [ds]


def _tidy(ds: MultiFieldList | FieldList, indent: int = 0) -> MultiFieldList | FieldList:
    """Tidies up a MultiFieldList or FieldList by removing empty sources.

    Parameters
    ----------
    ds : Union[MultiFieldList, FieldList]
        The dataset to tidy.
    indent : int, optional
        The indentation level. Defaults to 0.

    Returns
    -------
    Union[MultiFieldList, FieldList]
        The tidied dataset.
    """
    if isinstance(ds, MultiFieldList):

        sources = [s for s in _flatten(ds) if len(s) > 0]
        if len(sources) == 1:
            return sources[0]
        return MultiFieldList(sources)
    return ds
