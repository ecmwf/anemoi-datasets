# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest


@pytest.mark.parametrize(
    "select",
    [
        # An explicit, standalone "*" disables the default/required columns entirely.
        "*",
        # Whitespace around a standalone "*" is still treated as a wildcard.
        "  *  ",
    ],
)
def test_odb_sql_str_wildcard_select_disables_required_columns(select: str) -> None:
    from anemoi.datasets.create.sources.odb import odb_sql_str

    sql = odb_sql_str(
        start_datetime="20250101000000",
        end_datetime="20250101235959",
        select=select,
        where="",
        flavour={
            "date_column_name": "date",
            "time_column_name": "time",
            "latitude_column_name": "lat",
            "longitude_column_name": "lon",
        },
        required_columns=["obsvalue"],
    )

    assert sql == (
        "SELECT *, WHERE (timestamp(date, time) >= 20250101000000 "
        "AND timestamp(date, time) <= 20250101235959)"
    )


def test_odb_sql_str_select_containing_asterisk_substring_keeps_required_columns() -> (
    None
):
    """A "*" that is part of a larger SELECT expression (e.g. count(*)) must not be
    treated as a wildcard select, and required/default columns should still be added."""
    from anemoi.datasets.create.sources.odb import odb_sql_str

    sql = odb_sql_str(
        start_datetime="20250101000000",
        end_datetime="20250101235959",
        select="count(*)",
        where="",
        flavour={
            "date_column_name": "date",
            "time_column_name": "time",
            "latitude_column_name": "lat",
            "longitude_column_name": "lon",
        },
        required_columns=["obsvalue"],
    )

    assert sql == (
        "SELECT date, time, lat, lon, obsvalue, count(*), WHERE "
        "(timestamp(date, time) >= 20250101000000 AND timestamp(date, time) <= 20250101235959)"  # noqa: E501
    )


def test_pivot_obs_df_deduplicates_and_warns_on_duplicate_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import pandas as pd

    from anemoi.datasets.create.sources.odb import pivot_obs_df

    df = pd.DataFrame(
        {
            "date": [20250101, 20250101, 20250101],
            "varno": [1, 1, 2],
            "obsvalue": [10.0, 10.0, 20.0],
        }
    )

    with caplog.at_level("WARNING"):
        pivoted = pivot_obs_df(df, values=["obsvalue"], columns=["varno"])

    assert any("Duplicate rows" in record.message for record in caplog.records)
    assert len(pivoted) == 1
    assert pivoted.loc[0, "obsvalue_1"] == 10.0
    assert pivoted.loc[0, "obsvalue_2"] == 20.0


def test_pivot_obs_df_without_duplicates_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import pandas as pd

    from anemoi.datasets.create.sources.odb import pivot_obs_df

    df = pd.DataFrame(
        {
            "date": [20250101, 20250102],
            "varno": [1, 1],
            "obsvalue": [10.0, 30.0],
        }
    )

    with caplog.at_level("WARNING"):
        pivoted = pivot_obs_df(df, values=["obsvalue"], columns=["varno"])

    assert not any("Duplicate rows" in record.message for record in caplog.records)
    assert len(pivoted) == 2
