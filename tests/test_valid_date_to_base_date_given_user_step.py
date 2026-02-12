import datetime

import pytest

from anemoi.datasets.create.sources.mars import valid_date_to_base_date_given_user_step


def test_valid_date_to_base_date_given_user_step():
    first_date = datetime.datetime(2024, 1, 1, 0, 0)
    user_steps = [6, 12, 18, 24]
    # 0th valid date
    date = first_date + datetime.timedelta(hours=6)
    d, t, s = valid_date_to_base_date_given_user_step(date, user_steps, first_date)
    assert (d, t, s) == ("20240101", "0000", 6)
    # 1st valid date
    date = first_date + datetime.timedelta(hours=12)
    d, t, s = valid_date_to_base_date_given_user_step(date, user_steps, first_date)
    assert (d, t, s) == ("20240101", "0000", 12)
    # 4th valid date (second cycle, step wraps back to 6)
    date = first_date + datetime.timedelta(hours=30)
    d, t, s = valid_date_to_base_date_given_user_step(date, user_steps, first_date)
    assert (d, t, s) == ("20240102", "0000", 6)


def test_algorithm_uses_correct_step_index():
    """The step index must be derived from the cycle structure, not raw hours.

    For steps [6, 12, 18, 24] (cycle_length=24), all four valid dates
    in the first cycle must share the same base = first_date.
    The old algorithm used ``hours_since_first_date % len(steps)``
    which scrambles the mapping (e.g. +6 h gives index 6%4=2 → step 18
    instead of the correct step 6).
    """
    first_date = datetime.datetime(2024, 1, 1, 0, 0)
    user_steps = [6, 12, 18, 24]

    expected = [
        # (hours_offset, expected_base_date, expected_base_time, expected_step)
        (6, "20240101", "0000", 6),
        (12, "20240101", "0000", 12),
        (18, "20240101", "0000", 18),
        (24, "20240101", "0000", 24),
        # second cycle: base advances by 24 h
        (30, "20240102", "0000", 6),
        (36, "20240102", "0000", 12),
        (42, "20240102", "0000", 18),
        (48, "20240102", "0000", 24),
    ]

    for hours, exp_d, exp_t, exp_s in expected:
        date = first_date + datetime.timedelta(hours=hours)
        d, t, s = valid_date_to_base_date_given_user_step(date, user_steps, first_date)
        assert (d, t, s) == (
            exp_d,
            exp_t,
            exp_s,
        ), f"hours={hours}: got ({d}, {t}, {s}), expected ({exp_d}, {exp_t}, {exp_s})"


def test_first_date_none_raises():
    """Passing first_date=None must raise a clear error."""
    date = datetime.datetime(2024, 1, 1, 6, 0)
    with pytest.raises(ValueError, match="first_date"):
        valid_date_to_base_date_given_user_step(date, [6, 12], None)


if __name__ == "__main__":
    pytest.main([__file__])
