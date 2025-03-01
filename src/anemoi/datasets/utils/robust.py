# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import random
from datetime import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import requests

LOG = logging.getLogger(__name__)
# Code imported from ECMWF's `multiurl` package

RETRIABLE = (
    requests.codes.internal_server_error,
    requests.codes.bad_gateway,
    requests.codes.service_unavailable,
    requests.codes.gateway_timeout,
    requests.codes.too_many_requests,
    requests.codes.request_timeout,
)


def robust(
    call: Callable[..., requests.Response],
    maximum_tries: int = 500,
    retry_after: int = 120,
    mirrors: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> Callable[..., Optional[requests.Response]]:
    """A decorator to make HTTP requests robust by retrying on certain errors.

    Parameters
    ----------
    call : Callable[..., requests.Response]
        The function to call for making the HTTP request.
    maximum_tries : int, optional
        The maximum number of retry attempts (default is 500).
    retry_after : int, optional
        The number of seconds to wait before retrying (default is 120).
    mirrors : dict, optional
        A dictionary of mirrors to use for retrying the request.

    Returns
    -------
    Callable[..., Optional[requests.Response]]
        A wrapped function that retries the request on certain errors.
    """

    def retriable(code: int) -> bool:
        """Check if the HTTP status code is retriable.

        Parameters
        ----------
        code : int
            The HTTP status code.

        Returns
        -------
        bool
            True if the status code is retriable, False otherwise.
        """
        return code in RETRIABLE

    def wrapped(url: str, *args: Tuple, **kwargs: Dict) -> Optional[requests.Response]:
        """The wrapped function that retries the request on certain errors.

        Parameters
        ----------
        url : str
            The URL to make the request to.
        *args : tuple
            Additional positional arguments for the request.
        **kwargs : dict
            Additional keyword arguments for the request.

        Returns
        -------
        Optional[requests.Response]
            The HTTP response, or None if the request failed.
        """
        tries = 0
        main_url = url

        while tries < maximum_tries:
            try:
                r = call(main_url, *args, **kwargs)
            except requests.exceptions.SSLError:
                raise
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ) as e:
                r = None
                LOG.warning(
                    "Recovering from connection error [%s], attemps %s of %s",
                    e,
                    tries,
                    maximum_tries,
                )

            if r is not None:
                if not retriable(r.status_code):
                    return r
                LOG.warning(
                    "Recovering from HTTP error [%s %s], attemps %s of %s",
                    r.status_code,
                    r.reason,
                    tries,
                    maximum_tries,
                )

            tries += 1

            alternate = None
            replace = 0
            if mirrors is not None:

                for key, values in mirrors.items():
                    if url.startswith(key):
                        alternate = values
                        replace = len(key)
                        if not isinstance(alternate, (list, tuple)):
                            alternate = [alternate]

            if alternate is not None:
                mirror = random.choice(alternate)
                LOG.warning("Retrying using mirror %s", mirror)
                main_url = f"{mirror}{url[replace:]}"
            else:
                LOG.warning("Retrying in %s seconds", retry_after)
                time.sleep(retry_after)
                LOG.info("Retrying now...")

    return wrapped
