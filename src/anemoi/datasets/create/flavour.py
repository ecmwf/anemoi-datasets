# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
from anemoi.transform.fields import MISSING_METADATA
from anemoi.transform.flavour import RuleBasedFlavour as _RuleBasedFlavour
from anemoi.transform.metadata import key_to_component

# Label set on the fields whose level a flavour set (see ``_name_level`` in ``gridded/result.py``)
FLAVOUR_LEVEL_LABEL = "labels.anemoi_flavour_level"


class RuleBasedFlavour(_RuleBasedFlavour):
    """The rule-based flavour of anemoi-transform, recording on each field whether it set its level.

    The level a flavour sets can be the field's own GRIB level (e.g. ``levelist: 2`` on a 2 m temperature): the field
    alone does not tell it from a level that only went along with a new level type.
    """

    def apply(self, field: ekd.Field) -> ekd.Field:
        flavoured = super().apply(field)
        for key in self.keys():
            if key_to_component(key) == "vertical.level" and self(key, field) is not MISSING_METADATA:
                return flavoured.set(**{FLAVOUR_LEVEL_LABEL: True})
        return flavoured
