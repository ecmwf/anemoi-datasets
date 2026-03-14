# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import concurrent.futures
import json
import logging
import os
from typing import Any

LOG = logging.getLogger(__name__)


_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)


def _collect_analytics(event: str, **kwargs: Any) -> None:

    payload = dict(event=event, specific=kwargs, user=os.getlogin(), host=os.uname().nodename)

    LOG.info(f"Collecting analytics: {json.dumps(payload, indent=2)}")


def collect_analytics(event: str, **kwargs: Any) -> None:
    # Collect analytics data for a given event. This function is non-blocking and will return immediately.
    _executor.submit(_collect_analytics, event, **kwargs)
