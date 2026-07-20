# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Archive descriptions for the accumulate source — the ``from:`` block.

``period:`` says what the user wants; ``from:`` says what the source data
is.  There is no ``type:`` key — the shape is recognised structurally:

- ``base_dates`` + ``steps`` present — forecast runs (:class:`FromTrajectories`),
  described by ``base_dates`` (recurring initialisation times), ``steps``
  (the lead-time grid) and ``accumulation`` (``from-zero`` / a duration /
  ``from-zero-reset-every-<freq>``).  :class:`TrajectoryIntervalGenerator`
  turns such a description into candidate intervals for the covering search.
- ``lookup-table`` present — the explicit cycle lookup table
  (:class:`FromLookupTable`), the escape hatch for layouts that do not
  factorise (``LookupTableIntervalGenerator`` in ``interval_generators.py``).
- neither — a bare ``from:`` (:class:`FromBare`) stating only
  ``accumulation``.  In a ``layout: trajectories`` recipe it describes the
  run the layout imposes; in any other layout it describes base-less source
  data indexed by validity time alone, and ``accumulation`` is then the
  fixed length each field holds (handled in ``interval_generators.py``).

``from:`` is **optional**: omitting it recognises the description from a
well-known MARS source, which is the common case.  Recognition is the
default, not a value — there is no ``from: auto`` to write (the old spelling
is rejected with a message telling you to omit the key).  This holds in every
output layout: ``from:`` describes the *source data* and the output layout
decides the *output*, so an omitted ``from:`` is recognised the same way in a
``layout: trajectories`` recipe as anywhere else.

The older spellings (block-level ``accumulation:``, and
``covering:``/``availability:``) are accepted for one release with a
``DeprecationWarning``.  :func:`normalise_from` folds every
accepted spelling into the single ``from:`` union and is the *one*
implementation shared by recipe-time validation (:class:`AccumulateSchema`)
and the runtime source, so the two cannot drift apart.

This package also hosts the table of factorised descriptions for
well-known MARS archives used when ``from:`` is omitted.

The implementation is split across:

- :mod:`.accumulation` — the ``from.accumulation:`` scheme grammar;
- :mod:`.base_dates` — the recurring ``base_dates`` selectors;
- :mod:`.union` — the ``from:`` descriptions and the trajectory interval generator;
- :mod:`.mars_archives` — the recognised-archive table (omitted ``from:``);
- :mod:`.normalise` — the shared spelling-folding rules;
- :mod:`.schema` — the recipe-facing :class:`AccumulateSchema`.
"""

from .accumulation import ACCUMULATION_VALUES
from .accumulation import parse_accumulation
from .base_dates import WEEKDAY_NAMES
from .base_dates import WEEKDAYS
from .base_dates import RecurringBaseDates
from .mars_archives import _mars_archive_description
from .mars_archives import infer_from_trajectories
from .normalise import MIGRATE_HINT
from .normalise import _validate_from
from .normalise import check_valid_time_source
from .normalise import normalise_from
from .schema import AccumulateSchema
from .union import From
from .union import FromBare
from .union import FromLookupTable
from .union import FromTrajectories
from .union import TrajectoryIntervalGenerator

__all__ = [
    "ACCUMULATION_VALUES",
    "AccumulateSchema",
    "From",
    "FromBare",
    "FromLookupTable",
    "FromTrajectories",
    "MIGRATE_HINT",
    "RecurringBaseDates",
    "TrajectoryIntervalGenerator",
    "WEEKDAYS",
    "WEEKDAY_NAMES",
    # underscore-prefixed but part of the compat surface (migrate + tests import them)
    "_mars_archive_description",
    "_validate_from",
    "check_valid_time_source",
    "infer_from_trajectories",
    "normalise_from",
    "parse_accumulation",
]
