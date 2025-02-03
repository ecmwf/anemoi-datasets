# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from . import Command

LOG = logging.getLogger(__name__)


class Publish(Command):
    """Publish a dataset."""

    # This is a command that is used to publish a dataset.
    # it is a class, inheriting from Command.

    internal = True
    timestamp = True

    def add_arguments(self, parser):
        parser.add_argument("path", help="Path of the dataset to publish.")

    def run(self, args):
        try:
            from anemoi.registry import publish_dataset
        except ImportError:
            LOG.error("anemoi-registry is not installed. Please install it to use this command.")
            return

        publish_dataset(args.path)


command = Publish
