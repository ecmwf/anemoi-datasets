# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from .accumulations import range_parameter

LOG = logging.getLogger(__name__)


def execute(context, dates, use_cdsapi_dataset=None, **request):
    return range_parameter(context, dates, "minimum", use_cdsapi_dataset=use_cdsapi_dataset, **request)
