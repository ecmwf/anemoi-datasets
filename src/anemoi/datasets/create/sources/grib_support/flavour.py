# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict

from anemoi.transform.fields import Flavour
from anemoi.transform.fields import new_flavoured_field


class Rule:

    def __init__(self, rule):

        assert isinstance(rule, list)
        assert len(rule) == 2
        assert len(rule[0]) == 1

        self.key, self.value = list(rule[0].items())[0]
        self.rule = rule[1]

    def match(self, field):
        for k, v in self.rule.items():
            if field.metadata(k, default=None) != v:
                return False
        return True


class RuleBasedFlavour(Flavour):
    """Rule-based flavour for GRIB files."""

    def __init__(self, rules):
        assert isinstance(rules, list)
        self.rules = defaultdict(list)
        for rule in rules:
            rule = Rule(rule)
            self.rules[rule.key].append(rule)

    def apply(self, field):
        return new_flavoured_field(field, self)

    def __call__(self, key, field):
        """Called when the field metadata is queried"""

        for rule in self.rules[key]:
            if rule.match(field):
                return rule.value

        return field.metadata(key)
