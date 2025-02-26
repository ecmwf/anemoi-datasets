# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import threading
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

# from .dataset import FullIndex
# from .dataset import Shape
# from .dataset import TupleIndex

LOG = logging.getLogger(__name__)

_thread_local = threading.local()


DEFAULT_OPTIONS = dict(
    debug_zarr_loading=False,
)


def set_options(options: Optional[Union[bool, Dict]]) -> None:
    """Set options for opening datasets.

    Parameters
    ----------
    options : Optional[Union[bool, Dict]]
        Options for opening the dataset. If a boolean is provided, it will enable or disable
        certain default options. If a dictionary is provided, it should contain specific
        options as key-value pairs.
    """

    if not hasattr(_thread_local, "options"):
        _thread_local.options = DEFAULT_OPTIONS.copy()

    if options is None:
        _thread_local.options = DEFAULT_OPTIONS.copy()
        return

    if isinstance(options, bool):
        for key, value in DEFAULT_OPTIONS.items():
            if isinstance(value, bool):
                _thread_local.options[key] = options
        return

    _thread_local.options.update(options)

    print(json.dumps(_thread_local.options, indent=4))


def get_options() -> Dict:
    """Get the current options for opening datasets.

    Returns
    -------
    Dict
        The current options.
    """
    if not hasattr(_thread_local, "options"):
        _thread_local.options = DEFAULT_OPTIONS.copy()

    return _thread_local.options


def get_option(name: str, default=Optional[Any]) -> Optional[Any]:
    """Get the value of a specific option.

    Parameters
    ----------
    name : str
        The name of the option.

    default : Any, optional
        The default value to return if the option is not found. Defaults to None.

    Returns
    -------
    Optional[Any]
        The value of the option.
    """
    return get_options().get(name, default)
