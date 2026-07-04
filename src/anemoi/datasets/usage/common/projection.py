# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Projection machinery for usage-time origin tracking.

When a dataset is opened, it is usually wrapped in a chain of usage-time
views: ``subset`` (dates), ``select``/``drop`` (variables), ``join`` /
``concat`` (several datasets), ``thinning``, ``rescale``, ... To answer
"where does variable X of *this view* come from?", the view's indexing is
simulated and traced back to the underlying zarr store(s), where the
origin of every variable (its source and the filters applied at
dataset-create time) is recorded in the metadata.

The simulation works as follows:

1. ``dataset.components()`` builds a :class:`Projection` covering the
   whole view (one ``slice`` per dimension — for the gridded layout:
   ``(dates, variables, ensembles, grid)``) and calls
   ``dataset.project(projection)``.
2. Each wrapper's ``project()`` re-expresses the projection in the
   coordinates of the dataset it forwards to, and recurses:

   - index-changing wrappers (``subset``, ``select``) map the positions
     they keep with :meth:`Projection.from_indices`;
   - multi-dataset wrappers (``join``, ``concat``) split the projection,
     shifting the relevant axis into each constituent dataset with
     :meth:`Projection.offset`;
   - data-altering wrappers (``thinning``, ``rescale``, ``rename``, ...)
     leave the coordinates alone but record themselves with
     ``add_transformation``.

3. At the bottom of a chain, the zarr store binds the projection to
   itself (:meth:`ProjectionBase.from_store` →
   :class:`ProjectionStore`) and composes the accumulated slices with
   :meth:`ProjectionStore.apply`.

The final result is one :class:`ProjectionStore` per reached store region
(collected in a :class:`ProjectionList` when there are several), each
knowing which store it reads, which part of it, and which usage-time
transformations were traversed. :meth:`ProjectionStore.origins` then
takes the per-variable origin stored in the dataset metadata and pipes
onto it one ``{"when": "dataset-usage", "type": "filter", ...}`` step per
traversed transformation (as described by the transformation's
``origin_transformation`` method).

This machinery currently assumes the 4-dimensional gridded layout.
"""

import logging
from collections import defaultdict

LOG = logging.getLogger(__name__)


def _hashable(v):
    """Return a hashable (tuple-based) equivalent of a JSON-like structure.

    Origins are nested dicts/lists; converting them to sorted tuples lets
    them be used as dictionary keys when grouping variables that share an
    identical origin.
    """
    if isinstance(v, dict):
        return tuple((k, _hashable(vv)) for k, vv in sorted(v.items()))
    if isinstance(v, list):
        return tuple(_hashable(vv) for vv in v)
    return v


def _indices_to_slices(indices: list[int]) -> list[slice]:
    """Compress a set of indices into a minimal list of equivalent slices.

    Consecutive runs with a constant stride become one slice, e.g.
    ``[0, 2, 4, 5, 6]`` → ``[slice(0, 6, 2), slice(5, 7, 1)]``. Used by
    :meth:`Projection.from_indices` to express "the view keeps these
    positions" as slice arithmetic. The result is verified to expand back
    to exactly the input indices.

    Parameters
    ----------
    indices : list of int
        The positions kept by a view (duplicates are not allowed).

    Returns
    -------
    list of slice
        Slices that, concatenated, select exactly ``indices``.
    """
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


def _combine_slices(length, *slices) -> slice:
    """Compose successive slicing operations into a single slice.

    ``x[_combine_slices(len(x), s1, s2)]`` is ``x[s1][s2]``: each slice
    indexes the result of the previous one, and the composition is folded
    into one ``slice`` with a combined start and multiplied step. This is
    how a projection travelling down a chain of views accumulates the
    views' selections without materialising any data.

    Parameters
    ----------
    length : int
        The length of the axis the first slice applies to.
    *slices : slice
        The slices to compose, outermost first (all with non-negative
        bounds and positive steps).

    Returns
    -------
    slice
        The composed slice (the canonical empty slice ``slice(0, 0, 1)``
        when the composition selects nothing).
    """

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
    """Common behaviour of :class:`Projection`, :class:`ProjectionList` and
    :class:`ProjectionStore`.

    Wrappers manipulate projections only through this interface, so a
    single projection and a list of projections (produced when a ``join``
    or a non-contiguous selection splits the descent) can be handled
    uniformly.
    """

    def from_store(self, slices, store):
        """Bind a projection to a zarr store (the bottom of a view chain).

        Called by the store's ``project()`` with its own full-shape slices;
        the store then composes the incoming projection with
        :meth:`ProjectionStore.apply`.
        """
        return ProjectionStore(slices, store)

    @classmethod
    def from_slices(cls, slices):
        """Create a plain :class:`Projection` from per-dimension slices."""
        return Projection(slices)

    @classmethod
    def list_or_single(cls, projections):
        """Return the single projection, or wrap several in a :class:`ProjectionList`.

        Keeps the common case (one store region reached) free of list
        nesting while still supporting fan-out.
        """
        if len(projections) == 1:
            return projections[0]
        return ProjectionList(projections)

    def ensure_list(self):
        """Return self as an iterable of projections (see :class:`ProjectionList`)."""
        return ProjectionList([self])

    def compressed_origins(self) -> dict:
        """Return the origins of every variable, merged across projections.

        Returns
        -------
        dict
            ``{variable: [origin, ...]}`` — a list because the same
            variable name can be reached in several stores (e.g. through
            an overlay).
        """
        result = defaultdict(list)
        for p in self.ensure_list():
            for k, v in p.origins().items():
                result[k].append(v)
        return result

    def variables_origins(self) -> dict:
        """Return a compressed, indexed representation of the variable origins.

        Origins and dataset names are de-duplicated into two tables and
        every variable points into them by index, so datasets with many
        variables sharing a few origins serialise compactly.

        Returns
        -------
        dict
            ``{"datasets": [name, ...], "origins": [origin, ...],
            "variables": {variable: [origin_index, dataset_index]},
            "version": "1"}``.
        """
        origins = {}
        datasets = {}
        variables = {}
        for p in self.ensure_list():
            o = p.origins(compressed=True)
            name = p.dataset_name
            if name not in datasets:
                datasets[name] = len(datasets)
            dataset_index = datasets[name]

            for vars, origin in o.items():
                ho = _hashable(origin)
                if ho not in origins:
                    origins[ho] = [len(origins), origin]

                origin_index = origins[ho][0]

                for var in vars:
                    if var in variables:
                        # That is due to an overlay. To be fixed in the future
                        LOG.warning(f"Duplicate origin for {var}")
                    variables[var] = [origin_index, dataset_index]

        datasets = [k for k, _ in sorted(datasets.items(), key=lambda x: x[1])]
        origins = [o[1] for _, o in sorted(origins.items(), key=lambda x: x[1][0])]

        return dict(datasets=datasets, origins=origins, variables=variables, version="1")


class Projection(ProjectionBase):
    """Which part of a dataset a view selects: one ``slice`` per dimension.

    A projection starts as the full shape of the outermost view and is
    progressively re-expressed in the coordinates of each forwarded
    dataset while descending the chain of usage-time wrappers.
    """

    def __init__(self, slices):
        """Build a projection from per-dimension slices.

        Parameters
        ----------
        slices : list or tuple of slice
            One slice per dimension of the (gridded, 4-D) dataset:
            ``(dates, variables, ensembles, grid)``.
        """
        assert isinstance(slices, (list, tuple)), slices
        assert all(isinstance(s, slice) for s in slices), slices
        assert len(slices) == 4, slices
        self.slices = tuple(slices)

    def from_indices(self, *, axis, indices) -> "ProjectionBase":
        """Re-express the projection through a view that keeps ``indices``.

        Used by index-changing wrappers (``select`` on the variable axis,
        ``subset`` on the date axis): the wrapper presents positions
        ``0..n-1`` that map to ``indices`` in the forwarded dataset. The
        kept positions are compressed into slices
        (:func:`_indices_to_slices`) and each is composed with the current
        axis slice — first the index mapping, then this projection's own
        selection — yielding one projection per contiguous run.

        Parameters
        ----------
        axis : int
            The axis the view re-indexes.
        indices : list of int
            The positions of the forwarded dataset that the view keeps.

        Returns
        -------
        Projection or ProjectionList
            The projection(s) in the forwarded dataset's coordinates.
        """
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
        """Return a debug representation."""
        return f"Projection(slices={self.slices})"

    def offset(self, axis, amount) -> "Projection":
        """Return a copy with the ``axis`` slice shifted by ``amount``.

        Used by ``join``/``concat``: each constituent dataset occupies a
        contiguous block along the joined axis, so the projection is
        shifted by minus the block's offset to land in that dataset's own
        coordinates (positions outside it produce an empty composition
        downstream).

        Parameters
        ----------
        axis : int
            The axis to shift.
        amount : int
            The shift (negative to descend into a constituent dataset).

        Returns
        -------
        Projection
            The shifted projection.
        """
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
    """Several projections handled as one.

    Produced whenever the descent fans out — a ``join``/``concat`` sends
    the projection into each constituent dataset, and a non-contiguous
    selection splits one projection into several contiguous runs. All the
    :class:`ProjectionBase` operations distribute over the members;
    nested lists are flattened.
    """

    def __init__(self, projections):
        """Flatten ``projections`` (lists of lists) into a single list.

        Parameters
        ----------
        projections : list or tuple of ProjectionBase
            The projections to group.
        """
        assert isinstance(projections, (list, tuple)), projections
        assert all(isinstance(p, ProjectionBase) for p in projections), projections

        self.projections = []
        for p in projections:
            if isinstance(p, ProjectionList):
                self.projections.extend(p.projections)
            else:
                self.projections.append(p)

    def from_indices(self, *, axis, indices):
        """Distribute :meth:`Projection.from_indices` over the members."""
        return ProjectionList([p.from_indices(axis=axis, indices=indices) for p in self.projections])

    def __repr__(self):
        """Return a debug representation."""
        return "ProjectionList(" + ",".join(repr(p) for p in self.projections) + ")"

    def ensure_list(self):
        """Return self (already a list of projections)."""
        return self

    def __iter__(self):
        """Iterate over the member projections."""
        return iter(self.projections)

    def add_transformation(self, transformation):
        """Distribute ``add_transformation`` over the members."""
        return ProjectionList([p.add_transformation(transformation) for p in self.projections])


class ProjectionStore(ProjectionBase):
    """A projection resolved onto a zarr store: the end of the descent.

    Knows which store it reads, which region of it (``slices``) and which
    usage-time transformations were traversed on the way down
    (``transformations``, outermost first as registered by
    ``Forwards.project``). This is where the stored, dataset-create-time
    origins and the usage-time operations are combined.
    """

    def __init__(self, slices, store, transformations=None):
        """Bind a region of a store, with the traversed transformations.

        Parameters
        ----------
        slices : list or tuple of slice
            The selected region, in the store's own coordinates.
        store : ZarrStore
            The underlying store.
        transformations : list, optional
            The data-altering usage-time wrappers traversed while
            descending to this store.
        """
        assert isinstance(slices, (list, tuple)), slices
        assert all(isinstance(s, slice) for s in slices), slices
        assert len(slices) == 4, slices

        self.slices = slices
        self.store = store
        self.transformations = transformations or []

    def __repr__(self):
        """Return a debug representation."""
        return repr((self.slices, self.store.dataset_name))

    def apply(self, projection) -> "ProjectionBase":
        """Compose an incoming projection with this store's slices.

        Called by the store's ``project()``: ``self.slices`` covers the
        store's full shape and ``projection`` carries the selection
        accumulated by the views above; composing them axis by axis
        (:func:`_combine_slices`) yields the store region the view
        actually reads.

        Parameters
        ----------
        projection : ProjectionBase
            The projection(s) pushed down by the views.

        Returns
        -------
        ProjectionStore or ProjectionList
            One resolved store projection per incoming projection.
        """

        projections = projection.ensure_list()

        result = []

        for projection in projections:

            slices = []
            for a, b in zip(self.slices, projection.slices):
                slices.append(_combine_slices(a.stop, a, b))
            result.append(ProjectionStore(slices, self.store))

        return self.list_or_single(result)

    def variables(self):
        """Return the names of the variables selected by this projection."""
        return self.store.variables[self.slices[1]]

    def origins(self, compressed=False) -> dict:
        """Return the origin of each selected variable.

        For every variable in the projected region, the origin recorded in
        the store metadata at dataset-create time is extended with one
        step per traversed usage-time transformation: each transformation
        describes itself through ``origin_transformation(variable,
        origins)`` (``rename`` may also rename the variable, returning an
        ``(action, new_name)`` tuple). When there are usage-time steps the
        result is a ``{"type": "pipe", "when": "dataset-usage", "steps":
        [...]}`` structure whose first step is the stored origin.

        Parameters
        ----------
        compressed : bool
            When true, variables sharing an identical origin are grouped:
            the result maps *tuples* of variable names to one origin.

        Returns
        -------
        dict
            ``{variable: origin}`` (or ``{(variables, ...): origin}``
            when ``compressed``).
        """
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

        if compressed:

            compressed_result = defaultdict(list)
            for k, v in result.items():
                compressed_result[_hashable(v)].append((k, v))

            result = {}
            for v in compressed_result.values():
                key = tuple(sorted(k for k, _ in v))
                value = v[0][1]
                result[key] = value

        return result

    def add_transformation(self, transformation):
        """Return a copy with ``transformation`` appended to the traversal record."""
        return ProjectionStore(self.slices, self.store, self.transformations + [transformation])

    def __iter__(self):
        """Iterate over self (a single resolved projection)."""
        return iter([self])

    @property
    def dataset_name(self):
        """The name of the underlying store's dataset."""
        return self.store.dataset_name
