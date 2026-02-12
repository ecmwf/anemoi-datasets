# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from typing import Dict
from typing import List

import pytest

from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import covering_intervals
from anemoi.datasets.create.sources.accumulate_utils.interval_generators import interval_generator_factory


def build_signed_interval(x: str) -> SignedInterval:
    # x is like '20240101.0000 -> 20240101.0300, base=20240101.0000, ...'
    interval_part, *extras = x.split(",")
    interval_part = interval_part.strip()

    start_str, end_str = interval_part.split("->")
    start_str, end_str = start_str.strip(), end_str.strip()

    if len(start_str) == 15:
        start = datetime.strptime(start_str.strip(), "%Y%m%d %H:%M")
    else:
        start = datetime.strptime(start_str.strip(), "%Y%m%d.%H%M")

    if len(end_str) == 4:  # only hour provided, reuse date from start
        end_str = start.strftime("%Y%m%d") + "." + end_str

    if len(end_str) == 15:
        end = datetime.strptime(end_str, "%Y%m%d %H:%M")
    else:
        end = datetime.strptime(end_str, "%Y%m%d.%H%M")

    extras_dict = {}
    for extra in extras:
        key, value = extra.strip().split("=")
        if key == "base":
            extras_dict["base"] = datetime.strptime(value.strip(), "%Y%m%d.%H%M")
        else:
            raise ValueError(f"Unknown extra key: {key}")

    return SignedInterval(start=start, end=end, **extras_dict)


_ = build_signed_interval


ERA_TEST_CASES = [
    # era
    (
        _("20240101.1800 -> 20240101.2100"),
        [
            _("20240101.1800 -> 1900, base=20240101.1800"),
            _("20240101.1900 -> 2000, base=20240101.1800"),
            _("20240101.2000 -> 2100, base=20240101.1800"),
        ],
    ),
    (
        _("20240101.1800 -> 20240102.0600"),
        [
            _("20240101.1800 -> 1900, base=20240101.1800"),
            _("20240101.1900 -> 2000, base=20240101.1800"),
            _("20240101.2000 -> 2100, base=20240101.1800"),
            _("20240101.2100 -> 2200, base=20240101.1800"),
            _("20240101.2200 -> 2300, base=20240101.1800"),
            _("20240101.2300 -> 20240102.0000, base=20240101.1800"),
            _("20240102.0000 -> 0100, base=20240101.1800"),
            _("20240102.0100 -> 0200, base=20240101.1800"),
            _("20240102.0200 -> 0300, base=20240101.1800"),
            _("20240102.0300 -> 0400, base=20240101.1800"),
            _("20240102.0400 -> 0500, base=20240101.1800"),
            _("20240102.0500 -> 0600, base=20240101.1800"),
        ],
    ),
    (
        _("20240101.1800 -> 20240102.1200"),
        [
            _("20240101.1800 -> 1900, base=20240101.1800"),
            _("20240101.1900 -> 2000, base=20240101.1800"),
            _("20240101.2000 -> 2100, base=20240101.1800"),
            _("20240101.2100 -> 2200, base=20240101.1800"),
            _("20240101.2200 -> 2300, base=20240101.1800"),
            _("20240101.2300 -> 20240102.0000, base=20240101.1800"),
            _("20240102.0000 -> 0100, base=20240101.1800"),
            _("20240102.0100 -> 0200, base=20240101.1800"),
            _("20240102.0200 -> 0300, base=20240101.1800"),
            _("20240102.0300 -> 0400, base=20240101.1800"),
            _("20240102.0400 -> 0500, base=20240101.1800"),
            _("20240102.0500 -> 0600, base=20240101.1800"),
            _("20240102.0600 -> 0700, base=20240101.1800"),
            _("20240102.0700 -> 0800, base=20240101.1800"),
            _("20240102.0800 -> 0900, base=20240101.1800"),
            _("20240102.0900 -> 1000, base=20240101.1800"),
            _("20240102.1000 -> 1100, base=20240101.1800"),
            _("20240102.1100 -> 1200, base=20240101.1800"),
        ],
    ),
    (
        _("20240101.1800 -> 20240101.0600"),
        [
            _("20240101.1800 -> 20240101.1700, base=20240101.0600"),
            _("20240101.1700 -> 20240101.1600, base=20240101.0600"),
            _("20240101.1600 -> 20240101.1500, base=20240101.0600"),
            _("20240101.1500 -> 20240101.1400, base=20240101.0600"),
            _("20240101.1400 -> 20240101.1300, base=20240101.0600"),
            _("20240101.1300 -> 20240101.1200, base=20240101.0600"),
            _("20240101.1200 -> 20240101.1100, base=20240101.0600"),
            _("20240101.1100 -> 20240101.1000, base=20240101.0600"),
            _("20240101.1000 -> 20240101.0900, base=20240101.0600"),
            _("20240101.0900 -> 20240101.0800, base=20240101.0600"),
            _("20240101.0800 -> 20240101.0700, base=20240101.0600"),
            _("20240101.0700 -> 20240101.0600, base=20240101.0600"),
        ],
    ),
    (_("20240101.0000 -> 20240103.1515"), None),
    (_("20240101.1800 -> 20290103.0000"), None),
]

ENDA_TEST_CASES = [
    (_("20240102.0900 -> 20240102.1200"), [_("20240102.0900 -> 20240102.1200, base=20240102.0600")]),
    (_("20240102.0600 -> 20240102.0900"), [_("20240102.0600 -> 20240102.0900, base=20240102.0600")]),
    (
        _("20240102.1500 -> 20240102.2100"),
        [
            _("20240102.1500 -> 20240102.1800, base=20240102.0600"),
            _("20240102.1800 -> 20240102.2100, base=20240102.0600"),  # do not use base=20240102.1800
        ],
    ),
    (
        _("20240102.0900 -> 20240102.2100"),
        [
            _("20240102.0900 -> 20240102.1200, base=20240102.0600"),
            _("20240102.1200 -> 20240102.1500, base=20240102.0600"),
            _("20240102.1500 -> 20240102.1800, base=20240102.0600"),
            _("20240102.1800 -> 20240102.2100, base=20240102.0600"),
        ],
    ),
    (
        _("20240102.0900 -> 20240103.0600"),
        [
            _("20240102.0900 -> 20240102.1200, base=20240102.0600"),
            _("20240102.1200 -> 20240102.1500, base=20240102.0600"),
            _("20240102.1500 -> 20240102.1800, base=20240102.0600"),
            _("20240102.1800 -> 20240102.2100, base=20240102.0600"),
            _("20240102.2100 -> 20240103.0000, base=20240102.0600"),
            _("20240103.0000 -> 20240103.0300, base=20240102.1800"),
            _("20240103.0300 -> 20240103.0600, base=20240102.1800"),
        ],
    ),
    (
        _("20240102.0900 -> 20240103.2100"),
        [
            _("20240102.0900 -> 20240102.1200, base=20240102.0600"),
            _("20240102.1200 -> 20240102.1500, base=20240102.0600"),
            _("20240102.1500 -> 20240102.1800, base=20240102.0600"),
            _("20240102.1800 -> 20240102.2100, base=20240102.0600"),
            _("20240102.2100 -> 20240103.0000, base=20240102.0600"),
            _("20240103.0000 -> 20240103.0300, base=20240102.1800"),
            _("20240103.0300 -> 20240103.0600, base=20240102.1800"),
            _(
                "20240103.0600 -> 20240103.0900, base=20240102.1800"
            ),  # do not use base=20240103.0600, to avoid extra base change
            _(
                "20240103.0900 -> 20240103.1200, base=20240102.1800"
            ),  # do not use base=20240103.0600, to avoid extra base change
            _(
                "20240103.1200 -> 20240103.1500, base=20240103.0600"
            ),  # now do use base=20240103.0600 because step [21-24] is not available
            _("20240103.1500 -> 20240103.1800, base=20240103.0600"),
            _("20240103.1800 -> 20240103.2100, base=20240103.0600"),
        ],
    ),
    (
        _("20240102.0600 -> 20240102.1200"),
        [
            _("20240102.0600 -> 20240102.0900, base=20240102.0600"),
            _("20240102.0900 -> 20240102.1200, base=20240102.0600"),
        ],
    ),
]

GRIB_INDEX_TEST_CASE = [
    (
        _("20240102.0900 -> 20240102.1200"),
        [_("20240102.0900 -> 1000"), _("20240102.1000 -> 1100"), _("20240102.1100 -> 1200")],
    ),
]


RR_OPER_TEST_CASE = [
    (
        _("20240102.0900 -> 20240102.1200"),
        [_("20240102.0900 -> 0000, base=20240102.0000"), _("20240102.0000 -> 1200, base=20240102.0000")],
    ),
    (_("20240102.0000 -> 20240102.1200"), [_("20240102.0000 -> 1200, base=20240102.0000")]),
    (_("20240102.1900 -> 20240103.0000"), None),
    (_("20240102.0000 -> 20240102.2200"), None),
    (
        _("20240102.0000 -> 20240103.1200"),
        [_("20240102.0000 -> 20240103.0000, base=20240102.0000"), _("20240103.0000 -> 1200, base=20240103.0000")],
    ),
    (
        _("20240102.0000 -> 20240104.1200"),
        [
            _("20240102.0000 -> 20240103.0000, base=20240102.0000"),
            _("20240103.0000 -> 20240104.0000, base=20240103.0000"),
            _("20240104.0000 -> 20240104.1200, base=20240104.0000"),
        ],
    ),
]


class _Tester:
    def __init__(self, candidates):
        self.candidates = candidates

    def test(self, target: SignedInterval, expected: None | List[SignedInterval]):
        from rich import print

        start = target.start
        end = target.end
        print("=" * 60)
        actual = covering_intervals(start, end, self.candidates, error_on_fail=False)

        print("Target interval:", start, "→", end)
        if expected is None:
            assert actual is None, ("Expected no solution, but got one.", actual)
            print("No solution found as expected.")
            return
        if actual is None:
            assert False, ("Got no solution, but expected one.", expected)

        print("Solution intervals:")
        for it in actual:
            print(f"  {it}")
        print("Total intervals:", len(actual))
        print("Total covered (seconds):", sum(it.length for it in actual))
        print("Total absolute length (seconds):", sum(abs(it.length) for it in actual))
        print(
            "Base switches:",
            sum(1 for i in range(1, len(actual)) if actual[i].base != actual[i - 1].base),
        )

        if len(expected) != len(actual):
            print("--------------------------")
            for exp_it in expected:
                print(f"Expected: {exp_it}")
            print("--------------------------")
            for act_it in actual:
                print(f"Actual:   {act_it}")
            print("--------------------------")
        assert len(expected) == len(actual), (
            f"Expected {len(expected)} intervals, got {len(actual)}",
            expected,
            actual,
        )

        if not set(expected) == set(actual):
            print("--------------------------")
            print("Differences between expected and actual:")
            for exp_it, act_it in zip(expected, actual):
                flag = "✅" if exp_it == act_it else "❌"
                print(f"{flag} ", end="")
                print(exp_it, act_it)
                assert exp_it == act_it, f"Expected {exp_it}, got {act_it}"


@pytest.mark.parametrize("test", ERA_TEST_CASES, ids=[str(t[0]) for t in ERA_TEST_CASES])
def test_era(test):
    tester = _Tester(
        interval_generator_factory({"mars": {"class": "ea", "stream": "oper", "some_additional_key_to_be_ignored": 1}})
    )
    tester.test(test[0], test[1])


@pytest.mark.parametrize("test", ENDA_TEST_CASES, ids=[str(t[0]) for t in ENDA_TEST_CASES])
def test_enda(test):
    tester = _Tester(interval_generator_factory({"mars": {"class": "ea", "stream": "enda"}}))
    tester.test(test[0], test[1])


grib_index_config: Dict[int, str] = [
    # all period [i, i+1] are available
    (
        None,
        "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18/18-19/19-20/20-21/21-22/22-23/23-24",
    )
]


@pytest.mark.parametrize("test", GRIB_INDEX_TEST_CASE, ids=[str(t[0]) for t in GRIB_INDEX_TEST_CASE])
def test_grib_index_no_basetime(test):
    tester = _Tester(interval_generator_factory(grib_index_config))
    tester.test(test[0], test[1])


@pytest.mark.parametrize("test", RR_OPER_TEST_CASE, ids=[str(t[0]) for t in RR_OPER_TEST_CASE])
def test_rr_oper(test):
    tester = _Tester(interval_generator_factory({"mars": {"class": "rr", "stream": "oper", "origin": "se-al-ec"}}))
    tester.test(test[0], test[1])


# Config: bases every 18 h starting at 2024-01-01T00:00.
# Bases: 2024-01-01T00:00, 2024-01-01T18:00, 2024-01-02T12:00, ...
# Available steps: 0-6 h, 0-12 h, 0-18 h from each base.
_ABS_SEQ_CONFIG = [
    ({"start": "2024-01-01 00:00", "frequency": "18h"}, "0-6/0-12/0-18"),
]

ABSOLUTE_SEQUENCE_TEST_CASES = [
    # Single 18-h interval from first base.
    (
        _("20240101.0000 -> 20240101.1800"),
        [_("20240101.0000 -> 20240101.1800, base=20240101.0000")],
    ),
    # Single 18-h interval from second base.
    (
        _("20240101.1800 -> 20240102.1200"),
        [_("20240101.1800 -> 20240102.1200, base=20240101.1800")],
    ),
    # Single 6-h interval from first base.
    (
        _("20240101.0000 -> 20240101.0600"),
        [_("20240101.0000 -> 20240101.0600, base=20240101.0000")],
    ),
    # 6-h interval that requires subtraction: [0-12] minus [0-6] via negation.
    (
        _("20240101.0600 -> 20240101.1200"),
        [
            _("20240101.0600 -> 20240101.0000, base=20240101.0000"),  # negated [0-6]
            _("20240101.0000 -> 20240101.1200, base=20240101.0000"),  # [0-12]
        ],
    ),
    # 12-h interval from second base.
    (
        _("20240101.1800 -> 20240102.0600"),
        [_("20240101.1800 -> 20240102.0600, base=20240101.1800")],
    ),
    # Period not reachable from any base (24 h at an offset not covered).
    (
        _("20240101.0600 -> 20240101.1800"),
        [
            _("20240101.0600 -> 20240101.0000, base=20240101.0000"),  # negated [0-6]
            _("20240101.0000 -> 20240101.1800, base=20240101.0000"),  # [0-18]
        ],
    ),
]


@pytest.mark.parametrize("test", ABSOLUTE_SEQUENCE_TEST_CASES, ids=[str(t[0]) for t in ABSOLUTE_SEQUENCE_TEST_CASES])
def test_absolute_sequence(test):
    tester = _Tester(interval_generator_factory(_ABS_SEQ_CONFIG))
    tester.test(test[0], test[1])


# def with_reset_candidates(current_time: datetime, current_base:datetime, start:datetime, end:datetime) -> Iterable[SignedInterval]:
#     # Generate intervals similar to ERA but with reset frequency consideration
#
#     reset_frequency = 2  # days
#     bases = [current_base] if current_base is not None else []
#     bases += [datetime(current_time.year, current_time.month, current_time.day, h) for h in [6, 18]]
#     bases += [b - timedelta(days=1) for b in bases]
#     bases += [b + timedelta(days=1) for b in bases]
#
#     intervals = []
#     for base in bases:
#         # Check if the base aligns with the reset frequency
#         if (base.day - 1) % reset_frequency == 0:
#             for h in range(1, 19):
#                 start_interval = base
#                 end_interval = base + timedelta(hours=h)
#                 intervals.append(SignedInterval(start=start_interval, end=end_interval, base=base))
#     intervals = [i for i in intervals if i.start == current_time or current_time == i.end]
#
#     intervals = sorted(intervals, key=lambda x: -(x.base or x.start).timestamp())
#     return intervals
#
# WITH_RESET_CANDIDATES_TEST_CASE = [
#     # Test case where reset frequency is 2 days
#     (_("20240102.0900 -> 20240102.1200"), [_("20240102.0900 -> 1200, base=20240102.0600")]),
# ]
#
# @pytest.mark.parametrize("test", WITH_RESET_CANDIDATES_TEST_CASE, ids=[str(t[0]) for t in GRIB_INDEX_TEST_CASE])
# def test_with_reset_candidates(test):
#     tester = _Tester(with_reset_candidates)
#     tester.test(test[0], test[1])

if __name__ == "__main__":
    for t in ENDA_TEST_CASES:
        test_enda(t)
    for t in ERA_TEST_CASES:
        test_era(t)
    for t in GRIB_INDEX_TEST_CASE:
        test_grib_index_no_basetime(t)
    for t in RR_OPER_TEST_CASE:
        test_rr_oper(t)
