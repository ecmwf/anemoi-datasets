# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Layout-agnostic missing-date wrappers.

The classes are defined in :mod:`anemoi.datasets.usage.gridded.missing` and
operate on axis 0 only — both gridded ``(date, var, ens, cell)`` and
trajectory ``(base_date, var, ens, step, cell)`` arrays use that axis as
the time-like axis, so the same implementation applies.

Re-exporting from ``common`` lets the trajectory ``usage_factory_load``
discover :class:`MissingDates` and :class:`SkipMissingDates` (the gridded
copies are inside ``gridded/missing.py`` whose name does not match the
factory's lowercase symbol convention).
"""

from anemoi.datasets.usage.gridded.missing import MissingDates  # noqa: F401
from anemoi.datasets.usage.gridded.missing import SkipMissingDates  # noqa: F401
