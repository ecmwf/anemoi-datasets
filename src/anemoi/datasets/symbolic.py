# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.datasets.create.functions import all_filters
from anemoi.datasets.create.functions import all_sources


def _as_dict(obj):
    if isinstance(obj, dict):
        return {_as_dict(k): _as_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]

    if isinstance(obj, Result):
        return obj.as_dict()

    if obj == "class_":
        return "class"

    return obj


class Result:
    def __init__(self, obj, *args, **kwargs):
        self.obj = obj
        if args and kwargs:
            raise ValueError("Cannot have both args and kwargs")

        if len(args) == 1 and isinstance(args[0], dict):
            kwargs = args[0].copy()
            args = tuple()

        self.args = args
        self.kwargs = kwargs

    def __or__(self, other):
        return pipe(self, other)

    def __add__(self, other):
        return join(self, other)

    def as_dict(self):
        if self.args:
            return {self.obj.name: _as_dict(self.args)}
        else:
            return {self.obj.name: _as_dict(self.kwargs)}

    def __repr__(self):
        return f"{self.obj.name}({self.args}, {self.kwargs})"

    def update(self, *args, **kwargs):
        assert not self.args
        update = self.kwargs.copy()
        for arg in args:
            assert isinstance(arg, dict)
            update.update(**arg)
        update.update(**kwargs)

        return Result(self.obj, **update)


class JoinResult(Result):
    def __add__(self, other):
        return JoinResult(self.obj, *self.args, other)


class PipeResult(Result):
    def __add__(self, other):
        return PipeResult(self.obj, *self.args, other)


class SymbolicObject:
    ResultClass = Result

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.ResultClass(self, *args, **kwargs)


class SymbolicSource(SymbolicObject):
    pass


class SymbolicFilter(SymbolicObject):
    pass


class SymbolicJoin(SymbolicObject):

    ResultClass = JoinResult


class SymbolicPipe(SymbolicObject):
    ResultClass = PipeResult


for source, _ in all_sources():
    globals()[source] = SymbolicSource(source)

for filter, _ in all_filters():
    globals()[filter] = SymbolicFilter(filter)

pipe = SymbolicPipe("pipe")
join = SymbolicJoin("join")
repeated_dates = SymbolicSource("repeated_dates")
