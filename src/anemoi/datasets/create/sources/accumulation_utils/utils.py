# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any


class TodoList:
    def __init__(self, keys):
        self._todo = set(keys)
        self._len = len(keys)
        self._done = set()
        assert self._len == len(self._todo), (self._len, len(self._todo))

    def is_todo(self, key):
        return key in self._todo

    def is_done(self, key):
        return key in self._done

    def set_done(self, key):
        self._done.add(key)
        self._todo.remove(key)

    def all_done(self):
        if not self._todo:
            assert len(self._done) == self._len, (len(self._done), self._len)
            return True
        return False


def _member(field: Any) -> int:
    """Retrieves the member number from the field metadata.

    Parameters:
    ----------
    field : Any
        The field from which to retrieve the member number.

    Return:
    -------
    int
        The member number.
    """
    # Bug in eccodes has number=0 randomly
    number = field.metadata("number", default=0)
    if number is None:
        number = 0
    return number


def _to_list(x: list[Any] | tuple[Any] | Any) -> list[Any]:
    """Converts the input to a list if it is not already a list or tuple.

    Parameters:
    ----------
    x : Union[List[Any], Tuple[Any], Any]
        Input value.

    Return:
    -------
    List[Any]
        The input value as a list.
    """
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _scda(request: dict[str, Any]) -> dict[str, Any]:
    """Modifies the request stream based on the time.

    Parameters:
    ----------
    request : dict[str, Any]
        Request parameters.

    Return:
    -------
    dict[str, Any]
        The modified request parameters.
    """
    if request["time"] in (6, 18, 600, 1800):
        request["stream"] = "scda"
    else:
        request["stream"] = "oper"
    return request
