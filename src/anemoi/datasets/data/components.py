# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import cached_property

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
            # assert False, (length, slices)
            return slice(0, 0, 1)  # canonical empty slice

    if current_length == 0:
        # assert False, (length, slices)
        return slice(0, 0, 1)

    stop = start + current_length * step

    if step > 0 and stop > length:
        stop = length
    elif step < 0 and stop <= -1:
        stop = 0

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
        slices = indices_to_slices(indices)
        this_slice = self.slices[axis]
        combined = []
        for s in slices:
            # combined.append(combine_slices(max(this_slice.stop,s.stop), this_slice, s))
            combined.append(combine_slices(max(this_slice.stop, s.stop), s, this_slice))

        projections = [
            Projection([c if i == axis else self.slices[i] for i in range(len(self.slices))]) for c in combined
        ]

        if len(projections) == 1:
            return projections[0]
        else:
            return ProjectionList(projections)

    # def join(self, *, axis, shapes):
    #     assert isinstance(shapes, (list, tuple)), shapes
    #     assert all(isinstance(s, (list, tuple)) for s in shapes), shapes

    #     i = 0
    #     for s in shapes:
    #         i += s[axis]

    def advance(self, axis, amount):
        this_slice = self.slices[axis]
        new_start = this_slice.start + amount
        new_stop = this_slice.stop + amount
        slices = list(self.slices)
        slices[axis] = slice(new_start, new_stop, this_slice.step)
        return Projection(slices)

    def __repr__(self):
        return f"Projection(slices={self.slices})"


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

    # def join(self, *, axis, shapes):
    #     return ProjectionList([p.join(axis=axis, shapes=shapes) for p in self.projections])

    # def combine(self, *, axis, projections):
    #     assert False, projections

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
