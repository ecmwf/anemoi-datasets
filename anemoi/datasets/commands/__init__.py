#!/usr/bin/env python
# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import argparse
import importlib
import logging
import os
import sys

LOG = logging.getLogger(__name__)


def register(here, package, select, fail=None):
    result = {}
    not_available = {}

    for p in os.listdir(here):
        full = os.path.join(here, p)
        if p.startswith("_"):
            continue
        if not (p.endswith(".py") or (os.path.isdir(full) and os.path.exists(os.path.join(full, "__init__.py")))):
            continue

        name, _ = os.path.splitext(p)

        try:
            imported = importlib.import_module(
                f".{name}",
                package=package,
            )
        except ImportError as e:
            not_available[name] = e
            continue

        obj = select(imported)
        if obj is not None:
            result[name] = obj

    for name, e in not_available.items():
        if fail is None:
            pass
        if callable(fail):
            result[name] = fail(name, e)

    return result


class Command:
    def run(self, args):
        raise NotImplementedError(f"Command not implemented: {args.command}")


class Failed(Command):
    def __init__(self, name, error):
        self.name = name
        self.error = error

    def add_arguments(self, command_parser):
        command_parser.add_argument("x", nargs=argparse.REMAINDER)

    def run(self, args):
        print(f"Command '{self.name}' not available: {self.error}")
        sys.exit(1)


COMMANDS = register(
    os.path.dirname(__file__),
    __name__,
    lambda x: x.command(),
    lambda name, error: Failed(name, error),
)
