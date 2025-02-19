# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re
from abc import ABC
from abc import abstractmethod
from functools import wraps
from typing import Any
from typing import Callable
from typing import List
from typing import Union

LOG = logging.getLogger(__name__)


def notify_result(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result: Any = method(self, *args, **kwargs)
        self.context.notify_result(self.action_path, result)
        return result

    return wrapper


class Substitution(ABC):
    @abstractmethod
    def resolve(self, context: Any) -> Any:
        pass


class Reference(Substitution):
    def __init__(self, context: Any, action_path: List[str]) -> None:
        self.context: Any = context
        self.action_path: List[str] = action_path

    def resolve(self, context: Any) -> Any:
        return context.get_result(self.action_path)


def resolve(context: Any, x: Union[tuple, list, dict, Substitution, Any]) -> Any:
    if isinstance(x, tuple):
        return tuple([resolve(context, y) for y in x])

    if isinstance(x, list):
        return [resolve(context, y) for y in x]

    if isinstance(x, dict):
        return {k: resolve(context, v) for k, v in x.items()}

    if isinstance(x, Substitution):
        return x.resolve(context)

    return x


def substitute(context: Any, x: Union[tuple, list, dict, str, Any]) -> Any:
    if isinstance(x, tuple):
        return tuple([substitute(context, y) for y in x])

    if isinstance(x, list):
        return [substitute(context, y) for y in x]

    if isinstance(x, dict):
        return {k: substitute(context, v) for k, v in x.items()}

    if not isinstance(x, str):
        return x

    if re.match(r"^\${[\.\w]+}$", x):
        path: List[str] = x[2:-1].split(".")
        context.will_need_reference(path)
        return Reference(context, path)

    return x
