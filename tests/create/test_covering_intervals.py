from datetime import datetime
from datetime import timedelta
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import pytest

from anemoi.datasets.create.sources.covering_intervals import SignedInterval
from anemoi.datasets.create.sources.covering_intervals import covering_intervals


def build_signed_interval(x):
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


def era_func(current: datetime, current_base: Optional[datetime]) -> Iterable[SignedInterval]:
    # Define bases at 6:00 and 18:00 of the current day and the surrounding days
    # then provide intervals of lengths 1 to 18 hours for each base.

    bases = [current_base] if current_base is not None else []
    bases += [datetime(current.year, current.month, current.day, h) for h in [6, 18]]
    bases += [b - timedelta(days=1) for b in bases]
    bases += [b + timedelta(days=1) for b in bases]

    for base in bases:
        for h in range(1, 19):
            start = base
            end = base + timedelta(hours=h)
            if start <= current <= end:
                yield SignedInterval(start=start, end=end, base=base)


def enda_func(current: datetime, current_base: Optional[datetime]) -> Iterable[SignedInterval]:
    # Define bases at 6:00 and 18:00 of the current day and the surrounding days
    # then provide intervals of lengths 1 to 18 hours for each base.

    bases = [current_base] if current_base is not None else []
    bases += [b - timedelta(days=1) for b in bases]
    bases += [datetime(current.year, current.month, current.day, h) for h in [6, 18]]
    bases += [b + timedelta(days=1) for b in bases]

    for base in bases:
        for h in range(3, 19, 3):
            start = base + timedelta(hours=h)
            end = base + timedelta(hours=h + 3)
            if start <= current <= end:
                yield SignedInterval(start=start, end=end, base=base)


era_config: Dict[int, str] = {
    6: "0-1/0-2/0-3/0-4/0-5/0-6/0-7/0-8/0-9/0-10/0-11/0-12/0-13/0-14/0-15/0-16/0-17/0-18",
    18: "0-1/0-2/0-3/0-4/0-5/0-6/0-7/0-8/0-9/0-10/0-11/0-12/0-13/0-14/0-15/0-16/0-17/0-18",
}
enda_config: Dict[int, List[str]] = {
    6: "0-3/3-6/6-9/9-12/12-15/15-18",
    18: "0-3/3-6/6-9/9-12/12-15/15-18",
}
grib_index_config: Dict[int, str] = {
    # all period [i, i+1] are available
    None: "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18/18-19/19-20/20-21/21-22/22-23/23-24"
}


class Tester:
    def __init__(self, candidates):
        self.candidates = candidates

    def test(self, target: SignedInterval, expected: None | List[SignedInterval]):
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


_ = build_signed_interval


ERA_TEST_CASES = [
    # era
    (_("20240101.1800 -> 20240101.2100"), [_("20240101.1800 -> 2100, base=20240101.1800")]),
    (_("20240101.1800 -> 20240102.0600"), [_("20240101.1800 -> 20240102.0600, base=20240101.1800")]),
    (_("20240101.1800 -> 20240102.1200"), [_("20240101.1800 -> 20240102.1200, base=20240101.1800")]),
    (_("20240101.1800 -> 20240101.0600"), [_("20240101.1800 -> 20240101.0600, base=20240101.0600")]),
    (_("20240101.0000 -> 20240103.1515"), None),
    (_("20240101.1800 -> 20290103.0000"), None),
    (
        _("20240102.0000 -> 0300"),
        [
            _("20240102.0000 -> 20240101.1800, base=20240101.1800"),
            _("20240101.1800 -> 20240102.0300, base=20240101.1800"),
        ],
    ),
    (
        _("20240102.0900 -> 20240102.2100"),
        [
            _("20240102.0900 -> 20240102.0600, base=20240102.0600"),
            _("20240102.0600 -> 20240102.2100, base=20240102.0600"),
        ],
    ),
]


@pytest.mark.parametrize("test", ERA_TEST_CASES, ids=[str(t[0]) for t in ERA_TEST_CASES])
def test_era(test):
    tester = Tester(era_config)
    tester.test(test[0], test[1])


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


@pytest.mark.parametrize("test", ENDA_TEST_CASES, ids=[str(t[0]) for t in ENDA_TEST_CASES])
def test_enda(test):
    tester = Tester(enda_config)
    tester.test(test[0], test[1])


GRIB_INDEX_TEST_CASE = [
    (
        _("20240102.0900 -> 20240102.1200"),
        [_("20240102.0900 -> 1000"), _("20240102.1000 -> 1100"), _("20240102.1100 -> 1200")],
    ),
]


@pytest.mark.parametrize("test", GRIB_INDEX_TEST_CASE, ids=[str(t[0]) for t in GRIB_INDEX_TEST_CASE])
def test_grib_index_no_basetime(test):
    tester = Tester(grib_index_config)
    tester.test(test[0], test[1])


if __name__ == "__main__":
    for t in ENDA_TEST_CASES:
        test_enda(t)
    for t in ERA_TEST_CASES:
        test_era(t)
    for t in GRIB_INDEX_TEST_CASE:
        test_grib_index_no_basetime(t)
