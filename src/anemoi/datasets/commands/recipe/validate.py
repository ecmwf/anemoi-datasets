# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pydantic import ValidationError

from anemoi.datasets.create.recipe import Recipe


def validate_recipe(config: dict) -> str:
    try:
        Recipe(**config)
    except ValidationError as e:
        for error in e.errors():
            print(f"Field '{error['loc'][0]}': {error['msg']}.")
        raise
