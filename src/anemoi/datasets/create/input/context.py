# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
import textwrap

from anemoi.utils.dates import as_datetime as as_datetime
from anemoi.utils.dates import frequency_to_timedelta as frequency_to_timedelta
from anemoi.utils.humanize import plural

from anemoi.datasets.dates import DatesProvider as DatesProvider
from anemoi.datasets.fields import FieldArray as FieldArray
from anemoi.datasets.fields import NewValidDateTimeField as NewValidDateTimeField

from .trace import step
from .trace import trace

LOG = logging.getLogger(__name__)


class Context:
    def __init__(self):
        # used_references is a set of reference paths that will be needed
        self.used_references = set()
        # results is a dictionary of reference path -> obj
        self.results = {}

    def will_need_reference(self, key):
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        self.used_references.add(key)

    def notify_result(self, key, result):
        trace(
            "🎯",
            step(key),
            "notify result",
            textwrap.shorten(repr(result).replace(",", ", "), width=40),
            plural(len(result), "field"),
        )
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        if key in self.used_references:
            if key in self.results:
                raise ValueError(f"Duplicate result {key}")
            self.results[key] = result

    def get_result(self, key):
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        if key in self.results:
            return self.results[key]
        all_keys = sorted(list(self.results.keys()))
        raise ValueError(f"Cannot find result {key} in {all_keys}")
