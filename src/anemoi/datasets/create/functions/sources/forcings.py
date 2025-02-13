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

from anemoi.datasets.create.input.trace import support_fake_dates


def _fake_forcings(context, fake_dates, template, param):
    from anemoi.transform.fields import new_field_with_metadata

    provider = context.dates_provider
    real_dates = defaultdict(list)
    for date in fake_dates:
        real_date = provider.mapping[date]
        real_dates[real_date.valid_datetime].append(real_date)

    context.trace("✅", f"from_source(forcings, {template}, {param}")
    ds = from_source("forcings", source_or_dataset=template, date=sorted(real_dates.keys()), param=param)
    for f in ds:
        date = datetime.datetime.fromisoformat(f.metadata("valid_datetime"))
        assert date.year != 1900
        hindcasts = real_dates[date]
        for real_date in hindcasts:
            yield new_field_with_metadata(f, **real_date.metadata)


def fake_forcings(context, dates, template, param):
    from anemoi.transform.fields import new_fieldlist_from_list

    result = []
    for f in _fake_forcings(context, dates, template, param):
        result.append(f)

    return new_fieldlist_from_list(result)


@support_fake_dates(fake_forcings)
def forcings(context, dates, template, param):

    # from anemoi.datasets.dates import FakeDateProvider

    # provider = context.dates_provider
    # if isinstance(provider, FakeDateProvider):
    #     return fake_forcings(context, dates, template, param)

    context.trace("✅", f"from_source(forcings, {template}, {param}")
    return from_source("forcings", source_or_dataset=template, date=dates, param=param)


execute = forcings
