# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import concurrent.futures
import datetime
import json
import logging
import os
from threading import RLock
from typing import Any

from anemoi.utils.config import load_config as load_settings

LOG = logging.getLogger(__name__)
QUIET = set()


_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)


def analytics_rest_options():

    settings = load_settings()
    analytics = settings.get("analytics", {})
    url = analytics.get("url", "https://anemoi.ecmwf.int/api/v1/analytics")
    timeout = analytics.get("timeout", 5)

    return dict(url=url, timeout=timeout)


def _payload(event: str, **kwargs: Any) -> dict:
    from anemoi.datasets import __version__

    options = analytics_options()

    payload = dict(
        event=event,
        specific=kwargs,
        user=os.getlogin(),
        host=os.uname().nodename,
        anemoi_user=os.environ.get("ANEMOI_USER", options.get("anemoi_user", "unknown")),
        anemoi_datasets_version=__version__,
        time=datetime.datetime.utcnow().isoformat(),
    )
    return payload


COLLECT_ANALYTICS = None
LOCK = RLock()


def _do_collect_event(event: str, **kwargs: Any) -> None:
    global COLLECT_ANALYTICS, LOCK
    with LOCK:
        if COLLECT_ANALYTICS is not None:
            return COLLECT_ANALYTICS

        options = analytics_options()

        COLLECT_ANALYTICS = options.get("enabled", False)

        return COLLECT_ANALYTICS


def _collect_analytics_worker(event: str, **kwargs: Any) -> None:
    import requests

    global COLLECT_ANALYTICS

    try:

        if not _do_collect_event(event, **kwargs):
            return

        payload = _payload(event, **kwargs)
        config = analytics_rest_options()
        response = requests.post(config["url"], json=payload, timeout=config["timeout"])
        response.raise_for_status()
        if event not in QUIET:
            LOG.info(f"Analytics collected successfully for event: {event}")
            LOG.info("Use `anemoi-datasets analytics --disable` to stop collecting analytics.")
            QUIET.add(event)

    except Exception:
        LOG.exception("Failed to collect analytics")
        COLLECT_ANALYTICS = False


def analytics_options(options=None):
    path = os.path.expanduser("~/.config/anemoi/analytics.json")

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({}, f)

    if options is None:
        with open(path) as f:
            return json.load(f)
    else:
        with open(path, "w") as f:
            json.dump(options, f, indent=2)


def collect_analytics(event: str, print_analytics_only=False, **kwargs: Any) -> None:
    """Collect analytics data for a given event.
    This function is non-blocking and will return immediately.
    """

    if print_analytics_only:
        print(json.dumps(_payload(event, **kwargs), indent=2))
        return

    _executor.submit(_collect_analytics_worker, event, **kwargs)
