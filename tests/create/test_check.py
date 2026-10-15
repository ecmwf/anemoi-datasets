# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The validity table: which reduction may be applied to which archived statistic.

One case per cell, against the piece shapes that change the verdict.
"""

import datetime

import pytest

from anemoi.datasets.create.sources.windowed.check import Pieces
from anemoi.datasets.create.sources.windowed.check import check
from anemoi.datasets.create.sources.windowed.check import field_statistic

H = datetime.timedelta(hours=1)

#: Six hourly pieces, each held outright.
DIRECT = Pieces(lengths=frozenset({H}), differenced=None, fewest=6)
#: The CERRA / rr shape: hourly pieces, then a 3-hourly one.
MIXED = Pieces(lengths=frozenset({H, 3 * H}), differenced=None, fewest=4)
#: Six hourly pieces, each rebuilt as a(0,s+1) - a(0,s).
DIFFERENCED = Pieces(lengths=frozenset({H}), differenced="[6,7] = +a(0,7) -a(0,6)", fewest=6)
#: The whole window as one piece held outright.
ONE = Pieces(lengths=frozenset({6 * H}), differenced=None, fewest=1)

OK = None

CELLS = [
    # accumulate:
    ("sum", "accum", DIRECT, OK),
    ("sum", "accum", MIXED, OK),
    ("sum", "accum", DIFFERENCED, OK),
    ("sum", "accum", ONE, OK),
    ("sum", "max", DIRECT, "use 'maximum:'"),
    ("sum", "min", DIRECT, "use 'minimum:'"),
    ("sum", "avg", DIRECT, "use 'average:'"),
    # maximum: (minimum: is the mirror image)
    ("max", "accum", DIRECT, OK),
    ("max", "accum", DIFFERENCED, OK),
    ("max", "accum", MIXED, "different lengths"),
    ("max", "accum", ONE, "single piece"),
    ("max", "max", DIRECT, OK),
    ("max", "max", MIXED, OK),
    ("max", "max", ONE, OK),
    ("max", "max", DIFFERENCED, "not additive"),
    ("max", "min", DIRECT, "use 'minimum:'"),
    ("max", "avg", DIRECT, OK),
    ("max", "avg", MIXED, "same length"),
    ("max", "avg", DIFFERENCED, "not additive"),
    ("min", "accum", DIRECT, OK),
    ("min", "accum", ONE, "single piece"),
    ("min", "min", MIXED, OK),
    ("min", "max", DIRECT, "use 'maximum:'"),
    ("min", "avg", MIXED, "same length"),
    # average:
    ("mean", "accum", DIRECT, "divide by the period"),
    ("mean", "max", DIRECT, OK),
    ("mean", "max", MIXED, "same length"),
    ("mean", "max", DIFFERENCED, "not additive"),
    ("mean", "min", DIRECT, OK),
    ("mean", "min", MIXED, "same length"),
    ("mean", "avg", DIRECT, OK),
    ("mean", "avg", MIXED, OK),
    ("mean", "avg", DIFFERENCED, "m\\(6,9\\)"),
]


@pytest.mark.parametrize("operation,statistic,pieces,match", CELLS)
def test_the_table(operation: str, statistic: str, pieces: Pieces, match: str | None) -> None:
    if match is None:
        check(operation, statistic, pieces)
    else:
        with pytest.raises(ValueError, match=match):
            check(operation, statistic, pieces)


@pytest.mark.parametrize("operation", ["sum", "max", "min", "mean"])
@pytest.mark.parametrize("pieces", [DIRECT, MIXED, DIFFERENCED, ONE])
def test_an_unstated_statistic_is_accepted_on_trust(operation: str, pieces: Pieces) -> None:
    """GRIB1 says ``instant`` for genuine accumulations (``od`` ``cp``)."""
    check(operation, "instant", pieces)


class _Field:
    def __init__(self, time_method=None, step_type=None):
        self._time_method = time_method
        self._step_type = step_type

    def get(self, key):
        return self._time_method

    def metadata(self, key):
        if self._step_type is None:
            raise KeyError(key)
        return self._step_type


@pytest.mark.parametrize(
    "field,expected",
    [
        (_Field("max"), "max"),
        (_Field("instant", "accum"), "accum"),  # GRIB1: the parameter knows
        (_Field("instant"), "instant"),
        (_Field(None, None), "instant"),
    ],
)
def test_field_statistic(field, expected: str) -> None:
    assert field_statistic(field) == expected
