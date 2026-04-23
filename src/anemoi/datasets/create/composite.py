# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Composite ABC and Pipe utility for composing sources programmatically.

A Composite is a named factory for composite sources.  It carries its own
configuration; the source it wraps is supplied when the factory is called::

    class Accumulate(Composite):
        def __init__(self, period: timedelta):
            self.period = period

        def __call__(self, source: Source) -> Source:
            return AccumulateSource(source, self.period)

``Pipe`` applies a list of composites left-to-right, each wrapping the output
of the previous step::

    pipeline = Pipe(
        Mars(stream="oper", cls="od", param=["tp"], levtype="sfc"),
        [
            Accumulate(period=timedelta(hours=6)),
            Rename({"tp": "precip_6h"}),
        ],
    )
    result = pipeline.execute(ValidDates(dates))

Note: The ``Pipe`` here operates at the *programmatic* level — composites wrap
sources, and the typed argument flows through the whole chain.  The action-tree
``Pipe`` in ``input/action.py`` is a separate concept that dispatches YAML
config nodes; both patterns coexist.

Note: ``Composite`` is intentionally distinct from ``anemoi.transform.filter.Filter``,
which processes data flowing through a transform pipeline.  A ``Composite`` wraps
a ``Source``; an anemoi-transform ``Filter`` transforms a ``FieldList`` or
``DataFrame``.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from anemoi.datasets.create.source import Source


class Composite(ABC):
    """A named factory for composite sources.

    A Composite carries only its own configuration.  The source it wraps is
    supplied when the factory is called, making composites reusable and
    chainable via ``Pipe`` without resorting to lambdas.
    """

    @abstractmethod
    def __call__(self, source: Source) -> Source:
        """Wrap ``source`` in a composite source and return it.

        Parameters
        ----------
        source : Source
            The source to wrap.

        Returns
        -------
        Source
            A new composite source that delegates to ``source`` after
            (optionally) transforming the argument or post-processing
            the result.
        """


def Pipe(source: Source, composites: list[Composite]) -> Source:
    """Apply a sequence of composites left-to-right, returning the outermost source.

    Parameters
    ----------
    source : Source
        The innermost (leaf) source.
    composites : list[Composite]
        Composites applied in order.  Each wraps the source produced
        by the previous step.

    Returns
    -------
    Source
        The outermost composite source.  Calling ``execute(argument)`` on
        it runs the full pipeline.
    """
    for c in composites:
        source = c(source)
    return source
