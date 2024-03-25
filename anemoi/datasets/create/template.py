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
import textwrap
from functools import wraps

LOG = logging.getLogger(__name__)

TRACE_INDENT = 0


def step(action_path):
    return f"[{'.'.join(action_path)}]"


def trace(emoji, *args):
    print(emoji, " " * TRACE_INDENT, *args)


def trace_datasource(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        global TRACE_INDENT
        trace(
            "🌍",
            "=>",
            step(self.action_path),
            self._trace_datasource(*args, **kwargs),
        )
        TRACE_INDENT += 1
        result = method(self, *args, **kwargs)
        TRACE_INDENT -= 1
        trace(
            "🍎",
            "<=",
            step(self.action_path),
            textwrap.shorten(repr(result), 256),
        )
        return result

    return wrapper


def trace_select(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        global TRACE_INDENT
        trace(
            "👓",
            "=>",
            ".".join(self.action_path),
            self._trace_select(*args, **kwargs),
        )
        TRACE_INDENT += 1
        result = method(self, *args, **kwargs)
        TRACE_INDENT -= 1
        trace(
            "🍍",
            "<=",
            ".".join(self.action_path),
            textwrap.shorten(repr(result), 256),
        )
        return result

    return wrapper


def notify_result(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.context.notify_result(self.action_path, result)
        return result

    return wrapper


class Context:
    def __init__(self):
        # used_references is a set of reference paths that will be needed
        self.used_references = set()
        # results is a dictionary of reference path -> obj
        self.results = {}

    def will_need_reference(self, key):
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        self.used_references.add(key)

    def notify_result(self, key, result):
        trace("🎯", step(key), "notify result", result)
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        if key in self.used_references:
            if key in self.results:
                raise ValueError(f"Duplicate result {key}")
            self.results[key] = result

    def get_result(self, key):
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        if key in self.results:
            return self.results[key]
        all_keys = sorted(list(self.results.keys()))
        raise ValueError(f"Cannot find result {key} in {all_keys}")


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
