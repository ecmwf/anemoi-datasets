# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any

from anemoi.utils.registry import Registry

filter_registry = Registry(__name__)


def create_filter(context: Any, config: Any) -> Any:
    """Create a filter based on the provided configuration.

    Parameters
    ----------
    context : Any
        The context in which the filter is created.
    config : Any
        The configuration for the filter.

    Returns
    -------
    Any
        The created filter.
    """
    return filter_registry.from_config(config, context)
