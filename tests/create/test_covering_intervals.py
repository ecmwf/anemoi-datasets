# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Dict
from typing import List

import pytest
from interval_test_data import ENDA_TEST_CASES
from interval_test_data import ERA_TEST_CASES
from interval_test_data import GRIB_INDEX_TEST_CASE

from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import covering_intervals
from anemoi.datasets.create.sources.accumulate_utils.interval_generators import interval_generator_factory


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
    tester = _Tester(interval_generator_factory("era5-oper"))
    tester.test(test[0], test[1])


@pytest.mark.parametrize("test", ENDA_TEST_CASES, ids=[str(t[0]) for t in ENDA_TEST_CASES])
def test_enda(test):
    tester = _Tester(interval_generator_factory("era5-enda"))
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
