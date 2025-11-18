# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import codc as odc
import pandas

from anemoi.datasets.create.gridded.typing import DateList

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)


@source_registry.register("odb")
class OdbSource(Source):
    """ODB data source."""

    emoji = "🔭"

    def __init__(
        self,
        context,
        path: str,
        select: str | None = None,
        where: str | None = None,
        flavour: dict = {},
        pivot_columns: list = [],
        pivot_values: list = [],
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialise the ODB input.

        Parameters
        ----------
        context : dict
            The context.
        path : str
            The path to the ODB file.
        select : str, optional
            The select clause. Defaults to all columns ("*").
        where : str, optional
            The where clause. Defaults to no additional filtering ("").
        flavour : dict, optional
            Naming of the latitude, longitude, date and time columns. Defaults to
            {"latitude_column_name": "lat",
            "longitude_column_name": "lon",
            "date_column_name": "date",
            "time_column_name": "time"}.
        pivot_columns : list, optional
            List of column names - values in these columns will be used to
            define the new columns after the reshaping.
            Typically these identify entries in `pivot_values` as belonging to
            a particular observation type: for instance "channel_number" or
            "varno". Defaults to [].
        pivot_values : list, optional
            List of column names - values in these columns will
            be spread across different values of the columns. For instance,
            "observed_value" and "quality_control_value". Defaults to [].
        kwargs : dict, optional
            Additional keyword arguments.

        Note: All columns not specified in "pivot_columns" and "pivot_values"
            will be assumed to be "index" values (i.e. are the same within a
            given observation group).

        Note: Pivot values are named according to the unique values in the
            pivot columns. For instance, if
            `pivot_columns=["channel_number@body"]`
            with two unique channel numbers 1 and 2 that identify rows, and
            `pivot_values=["initial_obsvalue@body"]`, then the resulting columns
            will be named "observed_value_1" and "observed_value_2".

        """
        super().__init__(context)

        self.path = path
        if not select:
            select = "*"
            LOG.warning("No SELECT clause provided; defaulting to all columns.")
        if not where:
            where = ""
            LOG.warning("No WHERE clause provided; defaulting to no additional filtering.")
        self.select = select
        self.where = where
        self.flavour = {
            "latitude_column_name": "lat@hdr",
            "longitude_column_name": "lon@hdr",
            "date_column_name": "date@hdr",
            "time_column_name": "time@hdr",
        }
        self.flavour.update(flavour)
        self.pivot_columns = pivot_columns
        self.pivot_values = pivot_values

    def execute(self, dates: DateList) -> pandas.DataFrame:
        """Execute the ODB source.

        Parameters
        ----------
        dates : DateList
            The input dates.

        Returns
        -------
        pandas.dataframe.DataFrame
            The output dataframe.
        """
        start = dates[0].isoformat()
        end = dates[-1].isoformat()
        df = odb2df(
            start=start,
            end=end,
            path_str=self.path,
            select=self.select,
            where=self.where,
            flavour=self.flavour,
            pivot_columns=self.pivot_columns,
            pivot_values=self.pivot_values,
        )
        LOG.info(f"ODB source read {len(df)} rows from {self.path}")
        LOG.info(df)
        return df


def odb2df(
    start: str,
    end: str,
    path_str: str,
    select: str = "",
    where: str = "",
    pivot_columns: list = [],
    pivot_values: list = [],
    flavour: dict = {},
    keep_temp_odb: bool = False,
) -> pandas.DataFrame:
    """Read an ODB file using the given parameters and create a pandas DataFrame.

    Parameters
    ----------
    start : str
        Start datetime in ISO8601 format.
    end : str
        End datetime in ISO8601 format.
    path_str : str
        Path to the ODB file.
    select : str, optional
        SQL SELECT statement excluding the FROM clause and SELECT keyword.
    where : str, optional
        SQL WHERE clause excluding the WHERE keyword.
    pivot_columns : list, optional
        List of column names - values in these columns will be used to
        define the new columns after the reshaping.
        Typically these identify entries in `pivot_values` as belonging to
        a particular observation type: for instance "channel_number" or
        "varno".
    pivot_values : list, optional
        List of column names - values in these columns will
        be spread across different values of the columns. For instance,
        "observed_value" and "quality_control_value".
    flavour : dict, optional
        Naming of the latitude, longitude, date and time columns. Defaults to
        {"latitude_column_name": "lat",
        "longitude_column_name": "lon",
        "date_column_name": "date",
        "time_column_name": "time"}.
    keep_temp_odb : bool, optional
        Whether to keep the intermediate ODB file in temporary directory,
        default is False.

    Notes
    -----
    All columns not specified in "pivot_columns" and "pivot_values"
    will be assumed to be "index" values (i.e. are the same within a
    given observation group).

    Pivot values are named according to the unique values in the
    pivot columns. For instance, if
    `pivot_columns=["channel_number@body"]`
    with two unique channel numbers 1 and 2 that identify rows, and
    `pivot_values=["initial_obsvalue@body"]`, then the resulting columns
    will be named "observed_value_1" and "observed_value_2".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the parsed data.
    """
    path = Path(path_str)

    # Convert ISO8601 datetimes to YYYYMMDDHHMMSS
    start_datetime = iso8601_to_datetime(start)
    end_datetime = iso8601_to_datetime(end)
    LOG.info(f"Querying ODB file at {path} from {start_datetime} to {end_datetime}")

    sql = odb_sql_str(
        start_datetime,
        end_datetime,
        select,
        where,
        flavour,
        pivot_columns + pivot_values,
    )
    LOG.info(f"Using SQL query: {sql}")

    with tempfile.NamedTemporaryFile(suffix=".odb", delete=not keep_temp_odb) as intermediate_odb_path:
        subselect_odb_using_odc_sql(
            input_odb_path=path,
            output_odb_path=intermediate_odb_path.name,
            sql_query_string=sql,
        )
        df = odc.read_odb(intermediate_odb_path.name, single=True, aggregated=True)
        LOG.info(f"Intermediate ODB file created at: {intermediate_odb_path.name}")
    if keep_temp_odb:
        LOG.info(f"Intermediate ODB file kept at: {intermediate_odb_path.name}")

    assert isinstance(df, pandas.DataFrame)

    # The new "time" column has to be constructed from the existing date and
    # time columns which have them as YYYYMMDD and HHMMSS integers
    df["time"] = pandas.to_datetime(
        df[flavour["date_column_name"]].astype(str).str.zfill(8)
        + df[flavour["time_column_name"]].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
    )
    df.drop(columns=[flavour["date_column_name"], flavour["time_column_name"]], inplace=True)

    # The latitude and longitude columns may need renaming to standard names
    if flavour["latitude_column_name"] != "latitude":
        df.rename(
            columns={flavour["latitude_column_name"]: "latitude"},
            inplace=True,
        )
    if flavour["longitude_column_name"] != "longitude":
        df.rename(
            columns={flavour["longitude_column_name"]: "longitude"},
            inplace=True,
        )

    df_pivotted = pivot_obs_df(df, pivot_values, pivot_columns)
    return df_pivotted


def iso8601_to_datetime(iso8601_str: str) -> str:
    """Convert ISO8601 datetime string to YYYYMMDDHHMMSS string.

    Parameters
    ----------
    iso8601_str : str
        ISO8601 datetime string.

    Returns
    -------
    str
        Datetime string in YYYYMMDDHHMMSS format.
    """
    dt = datetime.fromisoformat(iso8601_str)
    return dt.strftime("%Y%m%d%H%M%S")


def odb_sql_str(
    start_datetime: str,
    end_datetime: str,
    select: str,
    where: str,
    flavour: dict,
    required_columns: list = [],
) -> str:
    """Construct an SQL query string for querying the ODB file.

    Parameters
    ----------
    start_datetime : str
        Start datetime in YYYYMMDDHHMMSS format.
    end_datetime : str
        End datetime in YYYYMMDDHHMMSS format.
    select : str
        SQL SELECT statement excluding the FROM clause and SELECT keyword.
    where : str
        SQL WHERE clause excluding the WHERE keyword.
    flavour : dict
        Dictionary containing specific options for the flavour.
    required_columns : list, optional
        List of required columns to include in the SELECT statement.

    Returns
    -------
    str
        Constructed SQL query string which will extract the required data with
        guaranteed columns given by the flavour along with any additional
        required columns. Others can be included via the `select` parameter.
    """
    date_col = flavour["date_column_name"]
    time_col = flavour["time_column_name"]
    lat_col = flavour["latitude_column_name"]
    lon_col = flavour["longitude_column_name"]

    required_columns = [col.strip() for col in required_columns]
    if select != "":
        if "*" in select:
            required_columns = []
        else:
            # Check for overlap between required_columns and select
            select_columns = [col.strip() for col in select.split(",")]  # Strip whitespace
            overlapping_columns = [col for col in required_columns if col in select_columns]
            if overlapping_columns:
                required_columns = [col for col in required_columns if col not in overlapping_columns]
            missing_columns = [col for col in required_columns if col not in overlapping_columns]
            if missing_columns:
                LOG.warning(
                    "Not all required columns are included in the "
                    f"SELECT statement. Missing columns: {missing_columns}"
                )

    default_select = f"{date_col}, {time_col}, {lat_col}, {lon_col}"
    if required_columns:
        default_select += ", " + ", ".join(required_columns)
    if select == "":
        select = default_select
    else:
        select = default_select + ", " + select

    # strip any trailing commas and whitespace - these are in the final query
    select = select.rstrip(", ").strip()

    # Note that whilst the where clause uses timestamp, outputting this directly
    # can cause large negative values instead of the expected YYYYMMDDHHMMSS
    # format.
    default_where = (
        f"(timestamp({date_col}, {time_col}) >= {start_datetime} "
        f"AND timestamp({date_col}, {time_col}) <= {end_datetime})"
    )
    if where == "":
        where = default_where
    else:
        where = default_where + f" AND ({where})"

    sql = f"SELECT {select}, WHERE {where}"

    return sql


def subselect_odb_using_odc_sql(
    input_odb_path: Path | str,
    output_odb_path: Path | str,
    sql_query_string: str,
) -> None:
    """Subselect ODB data based on an SQL query string using the ODC command-line tool
    and write to a new ODB file.

    Parameters
    ----------
    input_odb_path : Path
        Path to the input ODB file.
    output_odb_path : Path
        Path to the output ODB file.
    sql_query_string : str
        SQL query string for ODC.

    Raises
    ------
    FileNotFoundError
        If the input ODB file does not exist.
    RuntimeError
        If the ODC command in subprocess fails.
    """
    if not Path(input_odb_path).is_file():
        LOG.error(f"Input ODB file not found: {input_odb_path}")
        raise FileNotFoundError(f"Input ODB file not found: {input_odb_path}")

    command = [
        "odc",
        "sql",
        sql_query_string,
        "-i",
        input_odb_path,
        "-f",
        "odb",
        "-o",
        output_odb_path,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        LOG.info(f"Subsetted ODB written to: {output_odb_path}")

    except subprocess.CalledProcessError as e:
        LOG.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"ODC SQL command failed with exit code {e.returncode}") from e


def pivot_obs_df(df: pandas.DataFrame, values: list, columns: list) -> pandas.DataFrame:
    """Reshape the DataFrame, organized by the values in particular columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to pivot.
    values : list
        List of column names. Values in these columns will be spread across
        different values of the columns, such as observed value or quality
        control values.
    columns : list
        List of column names. Values in these columns will be used to define
        the new columns after reshaping, such as channel number or varno.

    Returns
    -------
    pandas.DataFrame
        Pivoted DataFrame.

    Notes
    -----
    The function reshapes the DataFrame based on the specified `columns` and
    `values`. All columns not specified in `columns` and `values` are assumed
    to be "index" values (i.e., they remain the same within a given observation
    group).

    For example, given the following DataFrame:

    +---+---+
    | A | B |
    +---+---+
    | 1 | 0.5 |
    | 2 | 0.1 |
    | 3 | 0.6 |
    | 1 | 0.3 |
    | 3 | 0.7 |
    +---+---+

    Using `columns=["A"]` and `values=["B"]`, the reshaped DataFrame would be:

    +-----+-----+-----+
    | B_1 | B_2 | B_3 |
    +-----+-----+-----+
    | 0.5 | 0.1 | 0.6 |
    | 0.3 | NaN | 0.7 |
    +-----+-----+-----+

    The column names in the resulting DataFrame are flattened to include the
    original column name and the unique values from the `columns` parameter.
    """
    # Calculate the index variables, based on all variables not in columns or values.
    indices = list(filter(lambda a: a not in values + columns, df.columns))
    # Perform the pivot
    pivoted = df.pivot(index=indices, columns=columns, values=values)
    # Flatten MultiIndex column names
    pivoted.columns = ["_".join(str(elem) for elem in col) for col in pivoted.columns]
    # Reset the dataframe index
    pivoted = pivoted.reset_index()
    return pivoted
