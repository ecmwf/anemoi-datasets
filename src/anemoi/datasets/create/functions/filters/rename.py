# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import re

from earthkit.data.indexing.fieldlist import FieldArray


class RenamedFieldMapping:
    """Rename a field based on the value of another field.

    Args:
        field (Field): The field to be renamed.
        what (str): The name of the field that will be used to rename the field.
        renaming (dict): A dictionary mapping the values of 'what' to the new names.
    """

    def __init__(self, field, what, renaming):
        self.field = field
        self.what = what
        self.renaming = renaming

    def metadata(self, key=None, **kwargs):
        if key is None:
            return self.field.metadata(**kwargs)

        value = self.field.metadata(key, **kwargs)
        if key == self.what:
            return self.renaming.get(self.what, {}).get(value, value)

        return value

    def __getattr__(self, name):
        return getattr(self.field, name)

    def __repr__(self) -> str:
        return repr(self.field)
        return f"{self.field} -> {self.what} -> {self.renaming}"


class RenamedFieldFormat:
    """Rename a field based on a format string.

    Args:
        format (str): A string that defines the new name of the field.
    """

    def __init__(self, field, format):
        self.field = field
        self.format = format
        self.bits = re.findall(r"{(\w+)}", format)

    def metadata(self, key, **kwargs):
        value = self.field.metadata(key, **kwargs)
        if "{" + key + "}" in self.format:
            bits = {b: self.field.metadata(b, **kwargs) for b in self.bits}
            return self.format.format(**bits)
        return value

    def __getattr__(self, name):
        return getattr(self.field, name)


def execute(context, input, what="param", **kwargs):
    if what in kwargs and isinstance(kwargs[what], str):
        return FieldArray([RenamedFieldFormat(fs, kwargs[what]) for fs in input])

    return FieldArray([RenamedFieldMapping(fs, what, kwargs) for fs in input])
