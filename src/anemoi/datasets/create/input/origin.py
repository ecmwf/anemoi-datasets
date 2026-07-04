# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Origins: the provenance of the variables of a dataset being created.

An :class:`Origin` records where a variable comes from: the source it was
loaded from (:class:`Source`), the filters applied to it
(:class:`Filter`, chained with :class:`Pipe`), and the merging of several
provenances (:class:`Join`).

The source and filter *actions* (see ``action.py``) each own one origin
object, built from the action's name and configuration. After an action
produces its result, ``Context.origin`` combines the action's origin with
whatever origin the data already carries (:meth:`Origin.combine`) and
tags the result — fields individually via the ``labels.anemoi_origin``
label, tabular frames as a whole via ``DataFrame.attrs``.

Origins compare and hash **by identity**, on purpose: all the data
produced by one action instance shares one origin *object*, so grouping
variables "by origin" (``_collect_origins``) means "produced by the same
source/filter chain", even when two actions happen to have identical
configurations. The caching in :meth:`Filter.combine` exists to preserve
this property for pipes.

At the end of the dataset creation the origins are serialised
(:meth:`Origin.as_dict`) into the ``origins`` entry of the dataset
metadata; operations applied later, at dataset-*usage* time, are appended
to these by the projection machinery (see
``anemoi.datasets.usage.common.projection``) — the ``when`` field tells
the two stages apart.
"""

import datetime
import logging
from abc import ABC

LOG = logging.getLogger(__name__)


class Origin(ABC):
    """The origin of a variable: the source it was loaded from and the
    filters applied to it while the dataset was created.

    Origins are attached to fields as the ``labels.anemoi_origin`` label
    and serialised (via :meth:`as_dict`) in the dataset metadata.
    """

    def __init__(self, when="dataset-create"):
        """Record when this origin applies (create vs usage time).

        Parameters
        ----------
        when : str
            ``"dataset-create"`` for origins built while creating a
            dataset; usage-time steps carry ``"dataset-usage"``.
        """
        self.when = when

    def __eq__(self, other):
        """Origins are equal only when they are the same object.

        Identity semantics are deliberate: "same origin" means "produced
        by the same action instance", not "same configuration".
        """
        if not isinstance(other, Origin):
            return False
        return self is other

    def __hash__(self):
        """Hash by identity (consistent with :meth:`__eq__`)."""
        return id(self)


def _un_dotdict(x):
    """Deep-copy a config into plain, JSON-serialisable Python types.

    Action configurations may contain ``DotDict``s, tuples, sets, dates or
    timedeltas; origins end up in the zarr attributes, which only accept
    JSON. Dates become ISO strings, timedeltas their ``str()`` form.
    """
    if isinstance(x, dict):
        return {k: _un_dotdict(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_un_dotdict(a) for a in x]

    if isinstance(x, (datetime.date, datetime.datetime)):
        return x.isoformat()

    if isinstance(x, datetime.timedelta):
        return str(x)

    return x


class Pipe(Origin):
    """A chain of processing steps: a source followed by filters.

    Built by :meth:`Filter.combine` when a filter processes data that
    already carries an origin — ``Pipe(mars, rename)`` reads "produced by
    the ``mars`` source, then transformed by the ``rename`` filter".
    """

    def __init__(self, s1, s2, when="dataset-create"):
        """Chain ``s2`` after ``s1``.

        When ``s1`` is itself a pipe, the new step is appended to its
        steps (pipes stay flat — a pipe of pipes is never built).

        Parameters
        ----------
        s1 : Origin
            The existing origin (a source, or a flat pipe).
        s2 : Origin
            The step to append (never a pipe).
        when : str
            See :class:`Origin`.
        """
        super().__init__(when)
        self.steps = [s1, s2]

        assert s1 is not None, (s1, s2)
        assert s2 is not None, (s1, s2)

        if isinstance(s1, Pipe):
            assert not isinstance(s2, Pipe), (s1, s2)
            self.steps = s1.steps + [s2]

    def combine(self, previous, action, action_arguments):
        """Never called: a pipe is produced by combination, it is not the
        origin declared by an action (only sources and filters are).
        """
        assert False, (self, previous)

    def as_dict(self):
        """Serialise as ``{"type": "pipe", "steps": [...], "when": ...}``."""
        return {
            "type": "pipe",
            "steps": [s.as_dict() for s in self.steps],
            "when": self.when,
        }

    def __repr__(self):
        """Return a debug representation (``source | filter | ...``)."""
        return " | ".join(repr(s) for s in self.steps)


class Join(Origin):
    """Several origins merged into one.

    Built when data with different provenances is combined — e.g. a
    filter whose inputs come from several sources
    (:meth:`Filter.combine`'s recovery path), or tabular frames from
    different sources concatenated by ``Context.join``.
    """

    def __init__(self, origins, when="dataset-create"):
        """Merge ``origins``.

        Parameters
        ----------
        origins : list, tuple or set of Origin
            The origins being merged.
        when : str
            See :class:`Origin`.
        """
        assert isinstance(origins, (list, tuple, set)), origins
        super().__init__(when)
        self.steps = list(origins)

        assert all(o is not None for o in origins), origins

    def combine(self, previous, action, action_arguments):
        """Never called: a join is produced by combination, it is not the
        origin declared by an action (only sources and filters are).
        """
        assert False, (self, previous)

    def as_dict(self):
        """Serialise as ``{"type": "join", "steps": [...], "when": ...}``."""
        return {
            "type": "join",
            "steps": [s.as_dict() for s in self.steps],
            "when": self.when,
        }

    def __repr__(self):
        """Return a debug representation (``origin & origin & ...``)."""
        return " & ".join(repr(s) for s in self.steps)


class Source(Origin):
    """The origin of data freshly produced by a source action (``mars``,
    ``grib``, ``forcings``, ``csv``, ...).

    Records the source's name and its full recipe configuration, so the
    dataset metadata tells exactly how to retrieve each variable again.
    """

    def __init__(self, name, config, when="dataset-create"):
        """Record a source and its configuration.

        Parameters
        ----------
        name : str
            The source's name in the recipe (last element of the action
            path).
        config : dict
            The source's recipe configuration (deep-copied to
            JSON-serialisable types).
        when : str
            See :class:`Origin`.
        """
        super().__init__(when)
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)

    def combine(self, previous, action, action_arguments) -> "Origin":
        """Return this source as the data's origin, overriding any inherited one.

        A source is where provenance starts, so ``previous`` is normally
        ``None``. It is not an error when it is not: fields built from a
        template (e.g. the ``forcings`` source, whose template is a
        ``${...}`` reference to another source's output) inherit the
        template's labels — including its origin — in earthkit-data 1.0.
        The source that actually produced the data wins; the template
        reference is visible in this source's config.

        Parameters
        ----------
        previous : Origin or None
            The origin the data already carries, if any.
        action : Action
            The source action (unused).
        action_arguments : Any
            The argument the action was called with (unused).

        Returns
        -------
        Origin
            Always ``self``.
        """
        if previous is not None:
            # Fields built from a template (e.g. the forcings source, whose
            # template is a ${...} reference to another source's output)
            # inherit the template's labels, including its origin. The source
            # that actually produced the fields is their origin; the template
            # reference is recorded in its config.
            LOG.debug(f"Source {self} overrides inherited origin {previous}")
        return self

    def as_dict(self):
        """Serialise as ``{"type": "source", "name": ..., "config": ..., "when": ...}``."""
        return {
            "type": "source",
            "name": self.name,
            "config": self.config,
            "when": self.when,
        }

    def __repr__(self):
        """Return a debug representation (name + object id, since identity matters)."""
        return f"{self.name}({id(self)})"


class Filter(Origin):
    """One processing step applied by a filter action (``rename``,
    ``regrid``, ``accumulate``'s inner filters, ...).

    A filter never *is* the full origin of a variable: combining it with
    the data's previous origin produces a :class:`Pipe` (see
    :meth:`combine`).
    """

    def __init__(self, name, config, when="dataset-create"):
        """Record a filter and its configuration.

        Parameters
        ----------
        name : str
            The filter's name in the recipe (last element of the action
            path).
        config : dict
            The filter's recipe configuration (deep-copied to
            JSON-serialisable types).
        when : str
            See :class:`Origin`.
        """
        super().__init__(when)
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)
        self._cache = {}

    def combine(self, previous, action, action_arguments) -> "Origin":
        """Pipe this filter onto the origin the data already carries.

        The normal path returns ``Pipe(previous, self)``. The pipe is
        **cached per** ``previous``: all the fields flowing through this
        filter from the same upstream get the *same* pipe object, which is
        what makes identity-based grouping of variables by origin work
        (see the module docstring).

        Recovery path: ``previous`` may be ``None`` when the filter
        produced fresh objects that did not inherit the origin tag (a
        user-plugin filter, or a tabular filter returning a brand-new
        DataFrame). The origin is then recovered from the filter's
        *input* (``action_arguments``): the single origin found there, or
        a :class:`Join` when the input mixes several. This recovery is
        also cached (per ``(action, input)``) and logged as a warning.

        Parameters
        ----------
        previous : Origin or None
            The origin the data already carries, if any.
        action : Action
            The filter action (used as a cache key in the recovery path).
        action_arguments : Any
            The filter's input — a field list, or a DataFrame for tabular
            pipelines — used to recover the origin when ``previous`` is
            ``None``.

        Returns
        -------
        Origin
            The combined origin (a :class:`Pipe`).
        """

        if previous is None:
            # This can happen if the filter does not tag its output with an origin
            # (e.g. a user plugin, or a tabular filter that returns a fresh
            # DataFrame without propagating ``attrs``). In that case we try to
            # get the origin from the action arguments.
            import pandas as pd

            key = (id(action), id(action_arguments))
            if key not in self._cache:

                LOG.warning(f"No previous origin to combine with: {self}. Action: {action}")
                LOG.warning(f"Connecting to action arguments {action_arguments}")
                origins = set()
                if isinstance(action_arguments, pd.DataFrame):
                    o = action_arguments.attrs.get("anemoi_origin")
                    if o is None:
                        raise ValueError("Cannot combine origins, previous is None and the input frame has no origin")
                    origins.add(o)
                else:
                    for k in action_arguments:
                        o = k.get("labels.anemoi_origin", default=None)
                        if o is None:
                            raise ValueError(
                                f"Cannot combine origins, previous is None and action_arguments {action_arguments} has no origin"
                            )
                        origins.add(o)
                if len(origins) == 1:
                    self._cache[key] = origins.pop()
                else:
                    self._cache[key] = Join(origins)
            previous = self._cache[key]

        if previous in self._cache:
            # We use a cache to avoid recomputing the same combination
            return self._cache[previous]

        self._cache[previous] = Pipe(previous, self)
        return self._cache[previous]

    def as_dict(self):
        """Serialise as ``{"type": "filter", "name": ..., "config": ..., "when": ...}``."""
        return {
            "type": "filter",
            "name": self.name,
            "config": self.config,
            "when": self.when,
        }

    def __repr__(self):
        """Return a debug representation (name + object id, since identity matters)."""
        return f"{self.name}({id(self)})"
