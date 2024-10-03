# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from functools import wraps

from anemoi.utils.dates import as_datetime as as_datetime
from anemoi.utils.dates import frequency_to_timedelta as frequency_to_timedelta
from earthkit.data.core.fieldlist import MultiFieldList
from earthkit.data.indexing.fieldlist import FieldList

from anemoi.datasets.dates import DatesProvider as DatesProvider
from anemoi.datasets.fields import FieldArray as FieldArray
from anemoi.datasets.fields import NewValidDateTimeField as NewValidDateTimeField

from ..functions import import_function

LOG = logging.getLogger(__name__)


def parse_function_name(name):

    if name.endswith("h") and name[:-1].isdigit():

        if "-" in name:
            name, delta = name.split("-")
            sign = -1

        elif "+" in name:
            name, delta = name.split("+")
            sign = 1

        else:
            return name, None

        assert delta[-1] == "h", (name, delta)
        delta = sign * int(delta[:-1])
        return name, delta

    return name, None


def is_function(name, kind):
    name, _ = parse_function_name(name)
    try:
        import_function(name, kind)
        return True
    except ImportError as e:
        print(e)
        return False


def assert_fieldlist(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        assert isinstance(result, FieldList), type(result)
        return result

    return wrapper


def assert_is_fieldlist(obj):
    assert isinstance(obj, FieldList), type(obj)


def _flatten(ds):
    if isinstance(ds, MultiFieldList):
        return [_tidy(f) for s in ds._indexes for f in _flatten(s)]
    return [ds]


def _tidy(ds, indent=0):
    if isinstance(ds, MultiFieldList):

        sources = [s for s in _flatten(ds) if len(s) > 0]
        if len(sources) == 1:
            return sources[0]
        return MultiFieldList(sources)
    return ds
