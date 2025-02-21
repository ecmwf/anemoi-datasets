# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from earthkit.data.indexing.fieldlist import FieldArray


class RenamedFieldMapping:
    """Rename a field based on the value of another field.

    Args:
        field (Any): The field to be renamed.
        what (str): The name of the field that will be used to rename the field.
        renaming (Dict[str, Dict[str, str]]): A dictionary mapping the values of 'what' to the new names.
    """

    def __init__(self, field: Any, what: str, renaming: Dict[str, Dict[str, str]]) -> None:
        """Initialize a RenamedFieldMapping instance.

        Args:
            field (Any): The field to be renamed.
            what (str): The name of the field that will be used to rename the field.
            renaming (Dict[str, Dict[str, str]]): A dictionary mapping the values of 'what' to the new names.
        """
        self.field = field
        self.what = what
        self.renaming = {}
        for k, v in renaming.items():
            self.renaming[k] = {str(a): str(b) for a, b in v.items()}

    def metadata(self, key: Optional[str] = None, **kwargs: Any) -> Any:
        """Get metadata from the original field, with the option to rename the parameter.

        Args:
            key (Optional[str]): The metadata key.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The metadata value.
        """
        if key is None:
            return self.field.metadata(**kwargs)

        value = self.field.metadata(key, **kwargs)
        if key == self.what:
            return self.renaming.get(self.what, {}).get(value, value)

        return value

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the original field.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute value.
        """
        return getattr(self.field, name)

    def __repr__(self) -> str:
        """Get the string representation of the original field.

        Returns:
            str: The string representation of the original field.
        """
        return repr(self.field)


class RenamedFieldFormat:
    """Rename a field based on a format string."""

    def __init__(self, field: Any, what: str, format: str) -> None:
        """Initialize a RenamedFieldFormat instance.

        Args:
            field (Any): The field to be renamed.
            what (str): The name of the field that will be used to rename the field.
            format (str): The format string for renaming.
        """
        self.field = field
        self.what = what
        self.format = format
        self.bits = re.findall(r"{(\w+)}", format)

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Get metadata from the original field, with the option to rename the parameter using a format string.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The metadata value.
        """
        value = self.field.metadata(*args, **kwargs)
        if args:
            assert len(args) == 1
            if args[0] == self.what:
                bits = {b: self.field.metadata(b, **kwargs) for b in self.bits}
                return self.format.format(**bits)
        return value

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the original field.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute value.
        """
        return getattr(self.field, name)

    def __repr__(self) -> str:
        """Get the string representation of the original field.

        Returns:
            str: The string representation of the original field.
        """
        return repr(self.field)


def execute(context: Any, input: List[Any], what: str = "param", **kwargs: Any) -> FieldArray:
    """Rename fields based on the value of another field or a format string.

    Args:
        context (Any): The context in which the function is executed.
        input (List[Any]): List of input fields.
        what (str, optional): The field to be used for renaming. Defaults to "param".
        **kwargs (Any): Additional keyword arguments for renaming.

    Returns:
        FieldArray: Array of renamed fields.
    """
    if what in kwargs and isinstance(kwargs[what], str):
        return FieldArray([RenamedFieldFormat(fs, what, kwargs[what]) for fs in input])

    return FieldArray([RenamedFieldMapping(fs, what, kwargs) for fs in input])
