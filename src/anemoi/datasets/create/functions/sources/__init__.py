# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import glob
import logging

from earthkit.data.utils.patterns import Pattern

LOG = logging.getLogger(__name__)


def _expand(paths):

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if path.startswith("file://"):
            path = path[7:]

        if path.startswith("http://"):
            yield path
            continue

        if path.startswith("https://"):
            yield path
            continue

        cnt = 0
        for p in glob.glob(path):
            yield p
            cnt += 1
        if cnt == 0:
            yield path


def iterate_patterns(path, dates, **kwargs):
    given_paths = path if isinstance(path, list) else [path]

    dates = [d.isoformat() for d in dates]
    if len(dates) > 0:
        kwargs["date"] = dates

    for path in given_paths:
        paths = Pattern(path, ignore_missing_keys=True).substitute(**kwargs)
        for path in _expand(paths):
            yield path, dates
