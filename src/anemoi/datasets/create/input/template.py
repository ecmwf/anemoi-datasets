# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import re
from functools import wraps

LOG = logging.getLogger(__name__)


def notify_result(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.context.notify_result(self.action_path, result)
        return result

    return wrapper


class Substitution:
    pass


class Reference(Substitution):
    def __init__(self, context, action_path):
        self.context = context
        self.action_path = action_path

    def resolve(self, context):
        return context.get_result(self.action_path)


def resolve(context, x):
    if isinstance(x, tuple):
        return tuple([resolve(context, y) for y in x])

    if isinstance(x, list):
        return [resolve(context, y) for y in x]

    if isinstance(x, dict):
        return {k: resolve(context, v) for k, v in x.items()}

    if isinstance(x, Substitution):
        return x.resolve(context)

    return x


def substitute(context, x):
    if isinstance(x, tuple):
        return tuple([substitute(context, y) for y in x])

    if isinstance(x, list):
        return [substitute(context, y) for y in x]

    if isinstance(x, dict):
        return {k: substitute(context, v) for k, v in x.items()}

    if not isinstance(x, str):
        return x

    if re.match(r"^\${[\.\w]+}$", x):
        path = x[2:-1].split(".")
        context.will_need_reference(path)
        return Reference(context, path)

    return x
