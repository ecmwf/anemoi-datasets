# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from collections import defaultdict

from earthkit.data import from_source


def _fake_forcings(context, dates, template, param):
    from anemoi.transform.fields import new_field_with_metadata

    provider = context.dates_provider
    newdates = defaultdict(list)
    for date in dates:
        hindcast = provider.mapping[date]
        newdates[hindcast.hdate + datetime.timedelta(hours=hindcast.step)].append(hindcast)

    context.trace("✅", f"from_source(forcings, {template}, {param}")
    ds = from_source("forcings", source_or_dataset=template, date=sorted(newdates.keys()), param=param)
    for f in ds:
        date = datetime.datetime.fromisoformat(f.metadata("valid_datetime"))
        hindcasts = newdates[date]
        for hindcast in hindcasts:
            yield new_field_with_metadata(f, hdate=hindcast.hdate, date=hindcast.refdate, step=hindcast.step, time=0)


def fake_forcings(context, dates, template, param):
    from anemoi.transform.fields import new_fieldlist_from_list

    result = []
    for f in _fake_forcings(context, dates, template, param):
        result.append(f)

    return new_fieldlist_from_list(result)


def forcings(context, dates, template, param):

    from anemoi.datasets.dates import FakeHindcastsDates

    provider = context.dates_provider
    if isinstance(provider, FakeHindcastsDates):
        return fake_forcings(context, dates, template, param)

    context.trace("✅", f"from_source(forcings, {template}, {param}")
    return from_source("forcings", source_or_dataset=template, date=dates, param=param)


execute = forcings
