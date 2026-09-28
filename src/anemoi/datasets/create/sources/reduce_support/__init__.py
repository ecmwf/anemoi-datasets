# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Windowed time reductions: the ``average``, ``minimum`` and ``maximum`` sources.

They share one implementation (:class:`~.source.ReduceSource`) and one recipe
shape, which is ``accumulate``'s: ``source:`` is where the data comes from,
``period:`` is the window wanted, ``from:`` is what the source data is.

.. code:: yaml

   average: {period: 24h, from: {frequency: 6h}, source: {mars: {...}}}
   minimum: {period: 24h, from: {frequency: 6h}, source: {mars: {...}}}
   maximum: {period: 24h, from: {frequency: 6h}, source: {mars: {...}}}

The reduction is named by the block, so there is no ``reduce:`` source and no
``operation:`` key in a recipe.

``from:`` has two shapes, recognised by whether ``base_dates`` is present:
base-less instantaneous fields (``{frequency: ...}``, either output layout), or
the forecast run a trajectory layout imposes (``{base_dates: true,
frequency: ...}``).
"""

from .description import FromInstants  # noqa: F401
from .description import FromRun  # noqa: F401
from .description import ReduceSchema  # noqa: F401
from .description import window_samples  # noqa: F401
from .source import AverageSource  # noqa: F401
from .source import MaximumSource  # noqa: F401
from .source import MinimumSource  # noqa: F401
from .source import ReduceSource  # noqa: F401
