# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

LOG = logging.getLogger(__name__)

OPTIONS = ("check_variables_compatibility",)


class Options:
    def __init__(self, parent: "Options" = None, **kwargs: Any) -> None:
        self._options = {} if parent is None else dict(parent._options)
        self._options.update(kwargs)

    @staticmethod
    def extract(args_or_kwargs: Any, options: Any = None) -> tuple[dict[str, Any], "Options"]:

        if options is None:
            options = Options()

        if not isinstance(args_or_kwargs, dict):
            return args_or_kwargs, options

        kwargs = dict(args_or_kwargs)
        options_dict = {}
        for option in OPTIONS:
            if option in kwargs:
                options_dict[option] = kwargs.pop(option)

        return kwargs, Options(parent=options, **options_dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self._options.get(key, default)

    def __repr__(self):
        return f"Options({self._options})"
