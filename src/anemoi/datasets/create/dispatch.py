# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Domain-named dispatch decorators for source execute() overloads.

Usage
-----
Decorate multiple ``execute`` overloads on a ``DispatchedSource`` subclass.
Each overload keeps the same method name but is tagged with the argument type
it handles.  The class-body namespace trick (``sys._getframe``) allows the
decorators to accumulate all overloads into a single ``_MultiDispatch``
descriptor despite the Python rule that each new ``def`` overwrites the
previous binding::

    from anemoi.datasets.create.dispatch import (
        DispatchedSource,
        for_valid_dates,
        for_forecast_dates,
        for_intervals,
        for_forecast_intervals,
    )

    class MySource(DispatchedSource):

        @for_valid_dates
        def execute(self, argument: ValidDates) -> FieldList:
            ...

        @for_intervals
        def execute(self, argument: Intervals) -> FieldList:
            ...

At call time the argument's concrete type (with MRO fallback) selects the
right overload.  Plain ``list[datetime]`` is wrapped in ``ValidDates``
automatically for backward compatibility.

Note
----
Sources that inherit from both ``LegacySource`` and ``DispatchedSource``
must list ``DispatchedSource`` *after* ``LegacySource`` in the MRO so that
``_MultiDispatch`` is not shadowed by ``LegacySource.execute``.  Alternatively
override ``execute`` explicitly as ``MarsSource`` does for Phase 0.
"""

from __future__ import annotations

import sys
from typing import Any

# ---------------------------------------------------------------------------
# Dispatch descriptor
# ---------------------------------------------------------------------------


class _MultiDispatch:
    """Method descriptor that dispatches ``execute()`` by argument type.

    Each instance carries its own ``_registry`` mapping ``argument_type →
    callable``.  Overloads are registered by the ``@for_*`` decorators at
    class-body time via frame inspection.

    At call time the argument's MRO is walked to find the best match.
    Plain ``list`` arguments are wrapped in ``ValidDates`` for backward compat.
    """

    def __init__(self, name: str = "execute") -> None:
        self.name = name
        self._registry: dict[type, Any] = {}

    def register(self, argument_type: type, fn: Any) -> "_MultiDispatch":
        """Register an overload for the given argument type.

        Parameters
        ----------
        argument_type : type
            The argument class this overload handles.
        fn : callable
            The unbound method to call.
        """
        self._registry[argument_type] = fn
        return self

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None):
        if obj is None:
            return self
        registry = self._registry
        obj_class_name = type(obj).__name__

        def dispatch(argument: Any) -> Any:
            from anemoi.datasets.create.arguments import ValidDates
            from anemoi.datasets.dates.groups import GroupOfDates

            # Backward compat: plain list[datetime] or GroupOfDates → ValidDates
            if isinstance(argument, list):
                argument = ValidDates(argument)
            elif isinstance(argument, GroupOfDates):
                argument = ValidDates(argument.dates)

            arg_type = type(argument)

            # Direct lookup
            method = registry.get(arg_type)
            if method is not None:
                return method(obj, argument)

            # MRO fallback — handles subclasses of registered types
            for arg_klass in arg_type.__mro__:
                method = registry.get(arg_klass)
                if method is not None:
                    return method(obj, argument)

            registered = [t.__name__ for t in registry]

            from anemoi.datasets.create.arguments import ForecastDates
            from anemoi.datasets.create.arguments import ForecastIntervals

            if isinstance(argument, (ForecastDates, ForecastIntervals)):
                raise NotImplementedError(
                    f"'{obj_class_name}' does not support the trajectory layout. "
                    f"Received {arg_type.__name__} but this source only handles: {registered}."
                )

            raise TypeError(
                f"{obj_class_name}.execute() has no overload for argument type "
                f"'{arg_type.__name__}'. Registered: {registered}"
            )

        return dispatch


# ---------------------------------------------------------------------------
# Decorator factory (frame-inspection approach)
# ---------------------------------------------------------------------------


def _make_for_decorator(argument_type: type):
    """Return a decorator that registers an execute overload for argument_type.

    Uses ``sys._getframe(1)`` to inspect the enclosing class body and
    accumulate multiple ``@for_*``-decorated methods with the same name into a
    single ``_MultiDispatch`` descriptor.

    Parameters
    ----------
    argument_type : type
        The argument class this overload handles (e.g. ``ValidDates``).
    """

    def decorator(fn: Any) -> _MultiDispatch:
        # Inspect the enclosing class body's local namespace.
        frame = sys._getframe(1)
        local_ns = frame.f_locals

        existing = local_ns.get(fn.__name__)
        if isinstance(existing, _MultiDispatch):
            dispatcher = existing
        else:
            dispatcher = _MultiDispatch(fn.__name__)

        dispatcher.register(argument_type, fn)
        return dispatcher

    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in argument_type.__name__).lstrip("_")
    decorator.__name__ = f"for_{snake}"
    return decorator


# ---------------------------------------------------------------------------
# Module-level decorators (lazy to avoid circular imports at load time)
# ---------------------------------------------------------------------------

_decorators: tuple | None = None


def _ensure_decorators() -> tuple:
    global _decorators
    if _decorators is None:
        from anemoi.datasets.create.arguments import ForecastDates
        from anemoi.datasets.create.arguments import ForecastIntervals
        from anemoi.datasets.create.arguments import Intervals
        from anemoi.datasets.create.arguments import ValidDates

        _decorators = (
            _make_for_decorator(ValidDates),
            _make_for_decorator(ForecastDates),
            _make_for_decorator(Intervals),
            _make_for_decorator(ForecastIntervals),
        )
    return _decorators


def __getattr__(name: str) -> Any:
    _names = ("for_valid_dates", "for_forecast_dates", "for_intervals", "for_forecast_intervals")
    if name in _names:
        return _ensure_decorators()[_names.index(name)]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# DispatchedSource mixin
# ---------------------------------------------------------------------------


class DispatchedSource:
    """Mixin that enables ``@for_*`` dispatch on ``execute()``.

    Subclasses define overloads by decorating methods named ``execute``
    with ``@for_valid_dates``, ``@for_intervals``, etc.  Because the
    decorators use frame inspection to accumulate overloads, multiple
    ``@for_*``-decorated ``def execute`` statements in the same class body
    all merge into a single ``_MultiDispatch`` descriptor::

        class MySource(Source, DispatchedSource):

            @for_valid_dates
            def execute(self, argument: ValidDates) -> FieldList:
                ...

            @for_intervals
            def execute(self, argument: Intervals) -> FieldList:
                ...

    The descriptor is placed on ``DispatchedSource`` itself and is inherited
    by all subclasses.  Per-class registries are stored on each subclass's
    own ``execute`` descriptor (placed there by the decorators).
    """

    pass
