# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import earthkit.data as ekd


def execute(context, input, **kwargs):
    # Useful to create a pipeline that returns an empty result
    # So we can reference an earlier step in a function like 'constants'
    return ekd.from_source("empty")
