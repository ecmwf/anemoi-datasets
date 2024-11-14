# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

from anemoi.utils.cli import Command
from anemoi.utils.cli import Failed
from anemoi.utils.cli import register_commands

__all__ = ["Command"]

COMMANDS = register_commands(
    os.path.dirname(__file__),
    __name__,
    lambda x: x.command(),
    lambda name, error: Failed(name, error),
)
