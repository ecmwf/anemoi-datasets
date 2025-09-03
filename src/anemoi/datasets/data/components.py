# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import cached_property

import rich
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
        assert s.stop >= s.start and s.step > 0
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

    return slice(start, stop, step)


class _Base:

    def from_store(self, slices, store):
        return ProjectionStore(slices, store)

    def make_new(self, slices):
        return Projection(slices)

    def list_or_single(self, projections):
        if len(projections) == 1:
            return projections[0]
        return ProjectionList(projections)

    def ensure_list(self):
        return ProjectionList([self])


class Projection(_Base):

    def __init__(self, slices):
        assert isinstance(slices, (list, tuple)), slices
        assert all(isinstance(s, slice) for s in slices), slices
        assert len(slices) == 4, slices
        self.slices = tuple(slices)

    def from_indices(self, *, axis, indices):
        length = max(indices) + 1
        slices = indices_to_slices(indices)
        this_slice = self.slices[axis]
        combined = []

        for s in slices:
            c = combine_slices(max(this_slice.stop, s.stop, length), s, this_slice)

            combined.append(c)

        projections = [
            Projection([c if i == axis else self.slices[i] for i in range(len(self.slices))]) for c in combined
        ]

        return self.list_or_single(projections)

    def from_slices(self, slices):
        return Projection(slices)

    def distribute(self, axis, shapes):

        rich.print("Distributing", self.slices[axis], [s[axis] for s in shapes])
        result = []
        sizes = [s[axis] for s in shapes]
        sizes = [sizes[0]] + [sizes[i] + sizes[i - 1] for i in range(1, len(sizes))]
        i = 0
        indices = []
        rich.print("Sizes", sizes)
        for indice in range(*self.slices[axis].indices(self.slices[axis].stop)):
            if i == len(sizes):
                break
            if indice < sizes[i]:
                indices.append(indice)
                continue

            if indices:
                for s in indices_to_slices(indices):
                    result.append(self.make_new([s if j == axis else self.slices[j] for j in range(len(self.slices))]))
                indices = []
            indices.append(indice)
            i += 1
        if indices:
            for s in indices_to_slices(indices):
                result.append(self.make_new([s if j == axis else self.slices[j] for j in range(len(self.slices))]))
        rich.print("======")
        for r in result:
            rich.print("Distributing", r)

        # for n in [s[axis] for s in shapes]:

        return self.list_or_single(result)

    def __repr__(self):
        return f"Projection(slices={self.slices})"

    def offset(self, axis, amount):
        return Projection(
            [slice(s.start + amount, s.stop + amount, s.step) if i == axis else s for i, s in enumerate(self.slices)]
        )

    def shape(self):
        return tuple(len(range(*s.indices(s.stop))) for s in self.slices)


class ProjectionList(_Base):
    def __init__(self, projections):
        assert isinstance(projections, (list, tuple)), projections
        assert all(isinstance(p, _Base) for p in projections), projections
        self.projections = []
        for p in projections:
            if isinstance(p, ProjectionList):
                self.projections.extend(p.projections)
            else:
                self.projections.append(p)

    def from_indices(self, *, axis, indices):
        return ProjectionList([p.from_indices(axis=axis, indices=indices) for p in self.projections])

    def distribute(self, axis, shapes):

        result = []
        offset = 0
        for p in self.projections:
            n = p.slices[axis].stop
            result.append(p.offset(axis, offset).distribute(axis=axis, shapes=shapes))
            offset += n

        return self.list_or_single(result)

        return ProjectionList([p.distribute(axis=axis, shapes=shapes) for p in self.projections])

    def __repr__(self):
        return "ProjectionList(" + ",".join(repr(p) for p in self.projections) + ")"

    def ensure_list(self):
        return self

    def __iter__(self):
        return iter(self.projections)


class ProjectionStore(_Base):
    def __init__(self, slices, store):
        assert isinstance(slices, (list, tuple)), slices
        assert all(isinstance(s, slice) for s in slices), slices
        assert len(slices) == 4, slices

        self.slices = slices
        self.store = store

    def __repr__(self):
        return repr((self.slices, self.store.dataset_name))

    def apply(self, projection):

        projections = projection.ensure_list()

        result = []

        for projection in projections:

            # rich.print('apply', projection, 'on', self)
            slices = []
            for a, b in zip(self.slices, projection.slices):
                slices.append(combine_slices(a.stop, a, b))
            result.append(ProjectionStore(slices, self.store))

        return self.list_or_single(result)


class Mapping:

    def __init__(self, slice: slice, length) -> None:
        self.slice = slice
        self.length = length

    def __repr__(self):
        return f"Mapping(slice={self.slice}, length={self.length})"

    def indices(self):
        return self.slice.indices(self.length)

    @property
    def start(self):
        return self.slice.start

    @cached_property
    def mapping(self):
        return {j: i for i, j in enumerate(range(*self.indices()))}


class Component:

    def reduce(self):
        result = []

        for slices, store in self._reduce():
            combined = []
            for i in range(len(slices)):
                s = combine_slices(store.shape[i], *slices[i])
                combined.append(Mapping(s, store.shape[i]))

            result.append((combined, store, slices))

        return result


class _ComponentList(Component):
    def __init__(self, components: list[Component], what, reason) -> None:
        self.components = components
        self.what = what
        self.reason = reason

    def __repr__(self):
        return "ComponentList(" + ",".join(repr(c) for c in self.components) + ")"

    def tree(self, tree=None):
        if tree is None:
            tree = Tree("Components")

        t = tree.add(f"{self.__class__.__name__}({self.what}, {self.reason})")
        for c in self.components:
            c.tree(t)
        return tree

    def _reduce(self):
        return sum([c._reduce() for c in self.components], [])


class Join(_ComponentList):
    # def _reduce(self):
    #     assert False, self.components
    pass


class Select(_ComponentList):
    pass


class Concat(_ComponentList):
    pass


class ZarrComponent(Component):
    def __init__(self, store) -> None:
        self.store = store

    def __repr__(self):
        return f"ZarrComponent({self.store})"

    def tree(self, tree=None):
        if tree is None:
            tree = Tree("Components")

        tree.add(f"ZarrComponent({self.store})")
        return tree

    def _reduce(self):
        slices = [[slice(0, s, 1)] for s in self.store.shape]
        return [(slices, self.store)]


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
        for slices, store in self.component._reduce():
            slices = slices.copy()
            slices[self.axis].append(self.slice)  # Add this slice to list
            result.append((slices, store))
        return result


class DateSpan(AxisComponent):
    axis = 0


class VariableSpan(AxisComponent):
    axis = 1
