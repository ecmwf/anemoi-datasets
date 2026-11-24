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

# Labels set on the fields whose level or level type a flavour set (see ``gridded/result.py``)
FLAVOUR_LEVEL_LABEL = "labels.anemoi_flavour_level"
FLAVOUR_LEVTYPE_LABEL = "labels.anemoi_flavour_levtype"
_LABELS = {"vertical.level": FLAVOUR_LEVEL_LABEL, "vertical.level_type": FLAVOUR_LEVTYPE_LABEL}


class RuleBasedFlavour(_RuleBasedFlavour):
    """The rule-based flavour of anemoi-transform, recording on each field whether it set its level and level type.

    The level or level type a flavour sets can be the field's own GRIB one (e.g. ``levelist: 2`` on a 2 m
    temperature, ``levtype: sfc`` on a surface field eccodes files under another levtype): the field alone does not
    tell them from the ones decoded from GRIB.
    """

    def apply(self, field: ekd.Field) -> ekd.Field:
        flavoured = super().apply(field)
        labels = {}
        for key in self.keys():
            label = _LABELS.get(key_to_component(key))
            if label is not None and self(key, field) is not MISSING_METADATA:
                labels[label] = True
        return flavoured.set(**labels) if labels else flavoured
