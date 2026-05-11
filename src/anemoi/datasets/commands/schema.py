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

from . import Command

LOG = logging.getLogger(__name__)


class Schema(Command):
    """Export the JSON schema of the recipe"""

    timestamp = False

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """
        pass

    def run(self, args: Any) -> None:
        import json

        from anemoi.datasets.create.recipe import Recipe

        print(json.dumps(Recipe.model_json_schema(), indent=2))


command = Schema
