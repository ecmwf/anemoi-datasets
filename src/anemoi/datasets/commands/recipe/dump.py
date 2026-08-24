# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json

from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.create.input.builder import InputBuilder
from anemoi.datasets.create.recipe import Recipe


class Dumper:

    def recipe(self, input, sources):
        return json.dumps({"input": input, "sources": sources}, indent=2, default=str)

    def call(self, name, config):
        return {name: config}

    def sources(self, sources):
        return sources

    def concat(self, actions):
        return {"concat": {"actions": actions}}

    def join(self, actions):
        return {"join": {"actions": actions}}

    def start_end_dates(self, start, end, frequency):
        return repr((str(start), str(end), frequency_to_string(frequency)))

    def steps(self, start, end, frequency):
        return repr((frequency_to_string(start), frequency_to_string(end), frequency_to_string(frequency)))


def dump_recipe(config: dict, dumper=None) -> str:
    recipe = Recipe(**config)
    input = InputBuilder(
        recipe.input,
        data_sources=recipe.data_sources,
    )
    if dumper is None:
        dumper = Dumper()

    return input.action.dump(dumper)
