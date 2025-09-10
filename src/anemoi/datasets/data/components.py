# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict


def _indices_to_slices(indices: list[int]) -> list[slice]:
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


def _combine_slices(length, *slices):

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


class ProjectionBase:

    def from_store(self, slices, store):
        return ProjectionStore(slices, store)

    @classmethod
    def from_slices(cls, slices):
        return Projection(slices)

    @classmethod
    def list_or_single(cls, projections):
        if len(projections) == 1:
            return projections[0]
        return ProjectionList(projections)

    def ensure_list(self):
        return ProjectionList([self])

    def compressed_origins(self):
        result = defaultdict(list)
        for p in self.ensure_list():
            for k, v in p.origins().items():
                result[k].append(v)
        return result


class Projection(ProjectionBase):

    def __init__(self, slices):
        assert isinstance(slices, (list, tuple)), slices
        assert all(isinstance(s, slice) for s in slices), slices
        assert len(slices) == 4, slices
        self.slices = tuple(slices)

    def from_indices(self, *, axis, indices):
        length = max(indices) + 1
        slices = _indices_to_slices(indices)
        this_slice = self.slices[axis]
        combined = []

        for s in slices:
            combined.append(_combine_slices(max(this_slice.stop, s.stop, length), s, this_slice))

        projections = [
            Projection([c if i == axis else self.slices[i] for i in range(len(self.slices))]) for c in combined
        ]

        return self.list_or_single(projections)

    def __repr__(self):
        return f"Projection(slices={self.slices})"

    def offset(self, axis, amount):
        return Projection(
            [
                (
                    slice(
                        s.start + amount,
                        s.stop + amount,
                        s.step,
                    )
                    if i == axis
                    else s
                )
                for i, s in enumerate(self.slices)
            ]
        )


class ProjectionList(ProjectionBase):
    def __init__(self, projections):
        assert isinstance(projections, (list, tuple)), projections
        assert all(isinstance(p, ProjectionBase) for p in projections), projections

        self.projections = []
        for p in projections:
            if isinstance(p, ProjectionList):
                self.projections.extend(p.projections)
            else:
                self.projections.append(p)

    def from_indices(self, *, axis, indices):
        return ProjectionList([p.from_indices(axis=axis, indices=indices) for p in self.projections])

    def __repr__(self):
        return "ProjectionList(" + ",".join(repr(p) for p in self.projections) + ")"

    def ensure_list(self):
        return self

    def __iter__(self):
        return iter(self.projections)

    def add_transformation(self, transformation):
        return ProjectionList([p.add_transformation(transformation) for p in self.projections])


class ProjectionStore(ProjectionBase):
    def __init__(self, slices, store, transformations=None):
        assert isinstance(slices, (list, tuple)), slices
        assert all(isinstance(s, slice) for s in slices), slices
        assert len(slices) == 4, slices

        self.slices = slices
        self.store = store
        self.transformations = transformations or []

    def __repr__(self):
        return repr((self.slices, self.store.dataset_name))

    def apply(self, projection):

        projections = projection.ensure_list()

        result = []

        for projection in projections:

            slices = []
            for a, b in zip(self.slices, projection.slices):
                slices.append(_combine_slices(a.stop, a, b))
            result.append(ProjectionStore(slices, self.store))

        return self.list_or_single(result)

    def variables(self):
        return self.store.variables[self.slices[1]]

    def origins(self):
        result = {}
        for variable in self.variables():

            origins = self.store.origins[variable]

            pipe = []
            for transformation in self.transformations:

                action = transformation.origin_transformation(variable, origins)
                if isinstance(action, tuple):
                    # Needed to support 'rename'
                    action, variable = action

                action = action.copy()
                action.setdefault("when", "dataset-usage")
                action.setdefault("type", "filter")
                pipe.append(action)

            if pipe:
                origins = {
                    "type": "pipe",
                    "when": "dataset-usage",
                    "steps": [origins] + pipe,
                }

            result[variable] = origins

        return result

    def add_transformation(self, transformation):
        return ProjectionStore(self.slices, self.store, self.transformations + [transformation])

    def __iter__(self):
        return iter([self])
