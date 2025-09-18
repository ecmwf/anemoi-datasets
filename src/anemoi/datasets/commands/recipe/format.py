# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging

from ...dumper import yaml_dump

LOG = logging.getLogger(__name__)


def make_dates(config):
    if isinstance(config, dict):
        return {k: make_dates(v) for k, v in config.items()}
    if isinstance(config, list):
        return [make_dates(v) for v in config]
    if isinstance(config, str):
        try:
            return datetime.datetime.fromisoformat(config)
        except ValueError:
            return config
    return config


ORDER = (
    "name",
    "description",
    "dataset_status",
    "licence",
    "attribution",
    "env",
    "dates",
    "common",
    "data_sources",
    "input",
    "output",
    "statistics",
    "build",
    "platform",
)


def format_recipe(args, config: dict) -> str:

    config = make_dates(config)
    assert config

    return yaml_dump(config, order=ORDER)
