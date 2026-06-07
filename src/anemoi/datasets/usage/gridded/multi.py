# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``multi``: read several named datasets through one shared two-step read.

``open_dataset(multi=dict(a=..., b=..., c=...))`` builds a :class:`Multi`.
Indexing it (``ds[n]``) returns a ``dict`` ``{name: array}`` — one array per
member, all for the same index.

The point of the container is **shared I/O**.  ``Multi`` implements
:meth:`collect_read_parts` / :meth:`read_from_buffer`, so the two-step read
gathers the leaf reads of *every* member into a single factorize/execute pass.
When several members read the same physical store — the classic case being one
global model used by several cutouts — that store is opened once (via
:func:`~anemoi.datasets.usage.store.shared_zarr_opens`) and read once, instead of
once per member.

See ``docs/adr/adr-3-two-step-read.md`` and ``docs/adr/two-step-read/``.
"""

import logging
from functools import cached_property
from typing import Any

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.forwards import Combined
from anemoi.datasets.usage.misc import _open
from anemoi.datasets.usage.store import shared_zarr_opens

LOG = logging.getLogger(__name__)


class Multi(Combined):
    """A named collection of datasets read through one shared pipeline.

    Indexing semantics
    ------------------
    ``ds[n]`` returns a ``{name: array}`` dict.  Two index forms:

    * **broadcast** — ``ds[date]`` / ``ds[date_slice]``: the same index is applied to
      every member.  Natural for the shared date axis (all members have the same
      length); returns each member's full field at that date.
    * **per-member dict** — ``ds[{name: index, ...}]``: each member is indexed by its
      *own* index.  This is the explicit, unambiguous way to grid-shard a multi
      (members have different grids — there is no single shared "multi grid")::

          ds[{"a": (t, slice(None), slice(None), shard_a),
              "b": (t, slice(None), slice(None), shard_b)}]

      Keys must be member names; a subset is allowed (only those members are read).
      I/O is still shared: all members' reads go through one factorize/execute pass,
      so a store common to several members is read once even with different shards.

    Parameters
    ----------
    datasets : dict of str to Dataset
        The named members.
    check_compatibility : bool, optional
        Whether to run the :class:`~anemoi.datasets.usage.forwards.Combined`
        compatibility checks across members.  Defaults to ``False`` because
        members are typically heterogeneous (different grids/variables, e.g.
        several different cutouts).
    """

    def __init__(self, datasets: dict[str, Dataset], check_compatibility: bool = False) -> None:
        assert isinstance(datasets, dict), datasets
        assert len(datasets) >= 1, "multi needs at least one member"

        self._check_compatibility = check_compatibility
        self.names = list(datasets.keys())
        self.datasets = [datasets[name].mutate() for name in self.names]
        self._members = dict(zip(self.names, self.datasets))

        lengths = {name: len(d) for name, d in zip(self.names, self.datasets)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"multi members must all have the same length, got {lengths}")

        if self._check_compatibility:
            for d in self.datasets[1:]:
                self.check_compatibility(self.datasets[0], d)

        # Forward scalar metadata (dates, frequency, …) to the first member.
        # NB: bypass Combined.__init__, which requires len > 1.
        self.forward = self.datasets[0]

    @property
    def members(self) -> dict[str, Dataset]:
        """The named members of the collection."""
        return dict(self._members)

    def mutate(self) -> Dataset:
        """Return the dataset unchanged."""
        return self

    def _member_indices(self, n: Any) -> dict[str, Any]:
        """Resolve an index to ``{name: per-member index}``.

        * ``dict`` — explicit **per-member** index (recommended for grid sharding):
          keys are member names, each value is that member's own full index, e.g.
          ``ds[{"a": (t, slice(None), slice(None), shard_a),
                "b": (t, slice(None), slice(None), shard_b)}]``.
          Keys must be members; a subset is allowed (only those members are read).
        * anything else — the same index is applied to every member (broadcast),
          the natural form for a shared-date read like ``ds[t]`` / ``ds[t0:t1]``.
        """
        if isinstance(n, dict):
            unknown = [k for k in n if k not in self._members]
            if unknown:
                raise KeyError(f"multi: unknown member key(s) {unknown}; members are {self.names}")
            return dict(n)
        return {name: n for name in self.names}

    # ------------------------------------------------------------------
    # Two-step read: gather every member's reads into one pass.
    # ------------------------------------------------------------------

    def collect_read_parts(self, n: FullIndex) -> "list | None":
        from anemoi.datasets.usage.read_parts import gather_parts

        idx = self._member_indices(n)
        return gather_parts(self._members[name].collect_read_parts(i) for name, i in idx.items())

    def read_from_buffer(self, n: FullIndex, buffer: Any) -> dict[str, Any]:
        idx = self._member_indices(n)
        return {name: self._members[name].read_from_buffer(i, buffer) for name, i in idx.items()}

    def __getitem__(self, n: FullIndex) -> dict[str, Any]:
        """Return ``{name: member[index]}`` (eager / fallback path)."""
        idx = self._member_indices(n)
        return {name: self._members[name][i] for name, i in idx.items()}

    # ------------------------------------------------------------------
    # Per-member metadata (keyed by name).
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.datasets[0])

    @cached_property
    def missing(self) -> set[int]:
        """The union of missing indices across all members."""
        result: set[int] = set()
        for d in self.datasets:
            result |= d.missing
        return result

    @property
    def shape(self) -> dict[str, Any]:
        """The shape of each member, keyed by name."""
        return {name: d.shape for name, d in zip(self.names, self.datasets)}

    @property
    def variables(self) -> dict[str, Any]:
        """The variables of each member, keyed by name."""
        return {name: d.variables for name, d in zip(self.names, self.datasets)}

    @property
    def statistics(self) -> dict[str, Any]:
        """The statistics of each member, keyed by name."""
        return {name: d.statistics for name, d in zip(self.names, self.datasets)}

    @property
    def latitudes(self) -> dict[str, Any]:
        """The latitudes of each member, keyed by name."""
        return {name: d.latitudes for name, d in zip(self.names, self.datasets)}

    @property
    def longitudes(self) -> dict[str, Any]:
        """The longitudes of each member, keyed by name."""
        return {name: d.longitudes for name, d in zip(self.names, self.datasets)}

    @property
    def name_to_index(self) -> dict[str, Any]:
        """The ``name_to_index`` mapping of each member, keyed by name."""
        return {name: d.name_to_index for name, d in zip(self.names, self.datasets)}

    @property
    def dtype(self) -> dict[str, Any]:
        """The dtype of each member, keyed by name."""
        return {name: d.dtype for name, d in zip(self.names, self.datasets)}

    # Heterogeneous members: the cross-member grid/variable checks do not apply.
    def check_same_grid(self, d1: Dataset, d2: Dataset) -> None:
        """No-op: members of a ``multi`` may live on different grids."""
        pass

    def check_same_variables(self, d1: Dataset, d2: Dataset) -> None:
        """No-op: members of a ``multi`` may have different variables."""
        pass

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Run compatibility checks only when explicitly requested."""
        if self._check_compatibility:
            super().check_compatibility(d1, d2)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Metadata specific to the subclass."""
        return {"names": self.names}

    def collect_supporting_arrays(self, collected: list[Any], *path: Any) -> None:
        """Collect each member's supporting arrays, namespaced by member name.

        Overrides the :class:`~anemoi.datasets.usage.forwards.Combined` default,
        which warns and keys by the member's own ``name`` (often ``None``).  Here
        the ``multi`` key is used as the path component, so e.g. a cutout's masks
        end up under that member's name.
        """
        for name, d in zip(self.names, self.datasets):
            d.collect_supporting_arrays(collected, *path, name)

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        """Return metadata, with members keyed by name under ``multi``."""
        return Dataset.metadata_specific(
            self,
            multi={name: d.metadata_specific() for name, d in zip(self.names, self.datasets)},
            **kwargs,
        )

    def tree(self) -> Node:
        """A tree representation of the named members."""
        return Node(self, [d.tree() for d in self.datasets], names=self.names)


def multi_factory(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Dataset:
    """Build a :class:`Multi` from ``multi=dict(name=spec, ...)``.

    All members are opened inside a single :func:`shared_zarr_opens` block, so a
    store referenced by more than one member (the same path string) is opened
    once and shared — which is what lets the two-step read merge their reads.

    Parameters
    ----------
    args : tuple
        Positional arguments (must be empty).
    kwargs : dict
        Keyword arguments.  ``multi`` is required; ``check_compatibility`` is
        optional.  No other options are accepted at the container level — put
        per-dataset options (``select``, ``start``, …) inside each member spec.
    """
    multi = kwargs.pop("multi")
    check_compatibility = kwargs.pop("check_compatibility", False)

    assert len(args) == 0, args
    assert isinstance(multi, dict), "multi must be a dict {name: dataset_spec}"
    assert not kwargs, (
        f"multi does not accept extra options {sorted(kwargs)}; "
        "put per-dataset options inside each member spec instead"
    )

    with shared_zarr_opens():
        members = {name: _open(spec) for name, spec in multi.items()}

    return Multi(members, check_compatibility=check_compatibility)
