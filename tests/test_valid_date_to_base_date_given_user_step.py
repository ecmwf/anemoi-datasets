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
    # 4th valid date (cycle)
    date = first_date + datetime.timedelta(hours=30)
    d, t, s = valid_date_to_base_date_given_user_step(date, user_steps, first_date)
    assert (d, t, s) == ("20240102", "0600", 6)


if __name__ == "__main__":
    pytest.main([__file__])
