# (C) Copyright 2024 Anemoi contributors.
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
import time
from threading import RLock
from typing import Any

from anemoi.utils.config import load_config as load_settings
from rich.console import Console
from rich.markdown import Markdown

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


def _collect_analytics(event: str, **kwargs: Any) -> None:
    import requests

    try:

        if not _collect_event_worker(event, **kwargs):
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


def _week(seconds: int) -> int:
    return seconds * 7 * 24 * 3600


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


def _collect_analytics_message(event, **kwargs) -> None:

    now = time.time()
    options = analytics_options()
    shown = options.get("message_shown", [])

    if shown:
        first = min(shown)
        last = max(shown)

        if now - first > _week(4):
            # Don't show ever again
            LOG.debug("Don't show analytics message ever again")
            return

        if now - last < _week(1):
            # Too soon
            LOG.debug("Don't show analytics message: too soon")
            return

    here = os.path.dirname(__file__)
    message = open(os.path.join(here, "analytics.md")).read()
    message = message.format(payload=json.dumps(_payload(event, **kwargs), indent=2))
    Console().print(Markdown(message))

    shown.append(now)
    options["message_shown"] = sorted(shown)
    analytics_options(options)


COLLECT_ANALYTICS = None
LOCK = RLock()


def _collect_event_worker(event: str, **kwargs: Any) -> None:
    global COLLECT_ANALYTICS, LOCK
    with LOCK:
        if COLLECT_ANALYTICS is not None:
            return COLLECT_ANALYTICS

        options = analytics_options()
        enabled = options.get("enabled")

        if enabled is None:
            _collect_analytics_message(event, **kwargs)
            COLLECT_ANALYTICS = False
        else:
            COLLECT_ANALYTICS = enabled

        return COLLECT_ANALYTICS


def collect_analytics(event: str, print_analytics_only=False, **kwargs: Any) -> None:
    """Collect analytics data for a given event.
    This function is non-blocking and will return immediately.
    """

    if print_analytics_only:
        print(json.dumps(_payload(event, **kwargs), indent=2))
        return

    _executor.submit(_collect_analytics, event, **kwargs)
