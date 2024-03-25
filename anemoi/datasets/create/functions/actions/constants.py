# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from climetlab import load_source


def constants(context, dates, template, param):
    context.trace("✅", f"load_source(constants, {template}, {param}")
    return load_source("constants", source_or_dataset=template, date=dates, param=param)


execute = constants
