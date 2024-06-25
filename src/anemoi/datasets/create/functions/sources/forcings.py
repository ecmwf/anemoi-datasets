# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from earthkit.data import from_source


def forcings(context, dates, template, param):
    context.trace("✅", f"from_source(forcings, {template}, {param}")
    return from_source("forcings", source_or_dataset=template, date=dates, param=param)


execute = forcings
