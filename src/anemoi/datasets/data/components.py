# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from rich.tree import Tree


def indices_to_slices(indices: list[int]) -> list[slice]:
    indices = sorted(indices)
    assert len(indices) == len(set(indices)), "Duplicate indices are not allowed"

    if not indices:
        return []

    slices = []
    n = len(indices)
    i = 0

    while i < n:
        start = indices[i]
        # default step = 1
        if i + 1 < n:
            step = indices[i + 1] - indices[i]
        else:
            step = 1

        j = i + 1
        while j < n and indices[j] - indices[j - 1] == step:
            j += 1

        stop = indices[j - 1] + step
        slices.append(slice(start, stop, step))
        i = j

    check = list()
    for s in slices:
        check.extend(range(s.start, s.stop, s.step))

    assert check == list(indices), slices

    return slices


def combine_slices(length, *slices):

    start, step, current_length = 0, 1, length

    for s in slices:
        new_start, new_stop, new_step = s.indices(current_length)
        new_length = len(range(new_start, new_stop, new_step))
        start = start + new_start * step
        step = step * new_step
        current_length = new_length

        if current_length == 0:
            return slice(0, 0, 1)  # canonical empty slice

    if current_length == 0:
        return slice(0, 0, 1)

    stop = start + current_length * step

    if step > 0 and stop > length:
        stop = None
    elif step < 0 and stop <= -1:
        stop = None

    return slice(start, stop, step)


class Component:

    def reduce(self):
        result = []

        for slices, name, shape in self._reduce():
            combined = []
            for i in range(len(slices)):
                combined.append(combine_slices(shape[i], *slices[i]))

            result.append((combined, name))

        return result


class ComponentList(Component):
    def __init__(self, components: list[Component]) -> None:
        self.components = components

    def __repr__(self):
        return "ComponentList(" + ",".join(repr(c) for c in self.components) + ")"

    def tree(self, tree=None):
        if tree is None:
            tree = Tree("Components")

        t = tree.add("ComponentList")
        for c in self.components:
            c.tree(t)
        return tree

    def _reduce(self):
        return sum([c._reduce() for c in self.components], [])


class ZarrComponent(Component):
    def __init__(self, name, shape) -> None:
        self.name = name
        self.shape = shape

    def __repr__(self):
        return f"ZarrComponent({self.name})"

    def tree(self, tree=None):
        if tree is None:
            tree = Tree("Components")

        tree.add(f"ZarrComponent({self.name})")
        return tree

    def _reduce(self):
        slices = [[slice(0, s, 1)] for s in self.shape]
        return [(slices, self.name, self.shape)]


class AxisComponent(Component):
    def __init__(self, slice, component) -> None:
        self.slice = slice
        self.component = component
        self.length = len(list(range(*self.slice.indices(self.slice.stop))))

    def __repr__(self):

        return f"{self.__class__.__name__}({self.slice} ({self.length}),{self.component})"

    def tree(self, tree=None):
        if tree is None:
            tree = Tree("Components")

        self.component.tree(tree.add(f"{self.__class__.__name__}({self.slice} ({self.length}))"))

        return tree

    def _reduce(self):
        result = []
        for slices, name, shape in self.component._reduce():
            slices = slices.copy()
            slices[self.axis].append(self.slice)  # Add this slice to list
            result.append((slices, name, shape))
        return result


class DateSpan(AxisComponent):
    axis = 0


class VariableSpan(AxisComponent):
    axis = 1
