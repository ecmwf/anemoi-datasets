# (C) Copyright 2024 Anemoi contributors.
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


def migrate_accumulations(config):
    """Migrate source 'accumulations' to the new 'accumulate' structure recursively."""
    if isinstance(config, dict):
        if "accumulations" in config:
            values = dict(config["accumulations"])
            period = values.pop("accumulation_period", 6)
            result = {k: migrate_accumulations(v) for k, v in config.items() if k != "accumulations"}
            result["accumulate"] = {
                "period": period,
                "availability": "auto",
                "source": {
                    "mars": values,
                },
            }
            return result
        return {k: migrate_accumulations(v) for k, v in config.items()}
    if isinstance(config, list):
        return [migrate_accumulations(item) for item in config]
    return config


def remove_useless_common_block(config):
    """Remove 'common' keys from the config."""
    return {k: v for k, v in config.items() if k != "common"}


def migrate(config: dict) -> dict:
    config = migrate_accumulations(config)
    config = remove_useless_common_block(config)
    return config


def migrate_recipe(args: Any, config) -> None:

    print(f"Migrating {args.path}")

    migrated = migrate(config)

    if migrated == config:
        return None

    return migrated
