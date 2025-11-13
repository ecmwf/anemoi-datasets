# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import codc
import pandas
from pathlib import Path
from datetime import datetime
import subprocess
import codc as odc
import tempfile

from anemoi.datasets.create.gridded.typing import DateList

from ..source import Source
from . import source_registry


@source_registry.register("odb")
class OdbSource(Source):
    """ODB data source."""

    emoji = "🔭"

    def __init__(
        self,
        context,
        path: str,
        select: str,
        where: str,
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
        select : str
            The select clause.
        where : str
            The where clause.
        flavour : dict, optional
            Naming of the latitude, longitude, date and time columns. Defaults to
            {"latitude column name": "lat",
            "longitude column name": "lon",
            "date column name": "date",
            "time column name": "time"}.
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

        Note: All columns not specified in "columns" and "values" will be
            assumed to be "index" values (i.e. are the same within a given
            observation group).
        Note: Pivot values are named according to the unique values in the
            pivot columns. For instance, if
            `pivot_columns=["channel_number@body"]`
            with two unique channel numbers 1 and 2 that identify rows, and
            `pivot_values=["initial_obsvalue@body"]`, then the resulting columns
            will be named "observed_value_1" and "observed_value_2".

        kwargs : dict, optional
            Additional keyword arguments.

        """
        super().__init__(context)

        self.path = path
        self.select = select
        self.where = where
        self.flavour = {
            "latitude column name": "lat",
            "longitude column name": "lon",
            "date column name": "date",
            "time column name": "time",
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
    """
    Read an ODB file using the given parameters and create a pandas DataFrame.

    Parameters:
    start (str): Start datetime (ISO8601 format)
    end (str): End datetime (ISO8601 format)
    path_str (str): Path to the ODB file
    select (str): SQL SELECT statement excluding FROM clause and SELECT keyword
    where (str): SQL WHERE clause excluding WHERE keyword
    pivot_columns (list, optional):
        List of column names - values in these columns will be used to
        define the new columns after the reshaping.
        Typically these identify entries in `pivot_values` as belonging to
        a particular observation type: for instance "channel_number" or
        "varno".
    pivot_values (list, optional):
        List of column names - values in these columns will
        be spread across different values of the columns. For instance,
        "observed_value" and "quality_control_value".
    keep_temp_odb (bool): Whether to keep the intermediate ODB file.

    Returns:
    pandas.DataFrame: DataFrame containing the parsed data.
    """
    path = Path(path_str)

    # Convert ISO8601 datetimes to YYYYMMDDHHMMSS
    start_datetime = iso8601_to_datetime(start)
    end_datetime = iso8601_to_datetime(end)
    print(f"Querying ODB file at {path} from {start_datetime} to {end_datetime}")

    sql = odb_sql_str(
        start_datetime,
        end_datetime,
        select,
        where,
        flavour,
        pivot_columns + pivot_values,
    )
    print(f"Using SQL query: {sql}")

    with tempfile.NamedTemporaryFile(
        suffix=".odb", delete=not keep_temp_odb
    ) as intermediate_odb_path:
        subselect_odb_using_odc_sql(
            input_odb_path=path,
            output_odb_path=intermediate_odb_path.name,
            sql_query_string=sql,
        )
        df = odc.read_odb(intermediate_odb_path.name, single=True, aggregated=True)
        print(f"Intermediate ODB file created at: {intermediate_odb_path.name}")
    if keep_temp_odb:
        print(f"Intermediate ODB file kept at: {intermediate_odb_path.name}")

    df_pivotted = pivot_obs_df(df, pivot_values, pivot_columns)

    return df_pivotted


def iso8601_to_datetime(iso8601_str: str) -> str:
    """
    Convert ISO8601 datetime string to YYYYMMDDHHMMSS string.

    Parameters:
    iso8601_str (str): ISO8601 datetime string.

    Returns:
    str: Datetime string in YYYYMMDDHHMMSS format.
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
    """
    Construct an SQL query string for querying the ODB file.

    Parameters:
    start_datetime (str): Start datetime in YYYYMMDDHHMMSS format
    end_datetime (str): End datetime in YYYYMMDDHHMMSS format
    select (str): SQL SELECT statement excluding FROM clause and SELECT keyword
    where (str): SQL WHERE clause excluding WHERE keyword
    flavour (dict): Dictionary containing specific options for the flavour.
    required_columns (list): List of required columns to include in the SELECT
        statement.

    Returns:
    str: Constructed SQL query string.
    """
    date_col = flavour.get("date column name", "date")
    time_col = flavour.get("time column name", "time")
    lat_col = flavour.get("latitude column name", "lat")
    lon_col = flavour.get("longitude column name", "lon")

    required_columns = [col.strip() for col in required_columns]
    if select != "":
        if "*" in select:
            required_columns = []
        else:
            # Check for overlap between required_columns and select
            select_columns = [
                col.strip() for col in select.split(",")
            ]  # Strip whitespace from select columns
            overlapping_columns = [
                col for col in required_columns if col in select_columns
            ]
            if overlapping_columns:
                required_columns = [
                    col for col in required_columns if col not in overlapping_columns
                ]
            missing_columns = [
                col for col in required_columns if col not in overlapping_columns
            ]
            if missing_columns:
                print(
                    "Warning: Not all required columns are included in the "
                    f"SELECT statement. Missing columns: {missing_columns}"
                )  # todo - switch to anemoi warning system.

    default_select = (
        f"timestamp({date_col}, {time_col}) as time, "
        f"{lat_col} as latitude, {lon_col} as longitude"
    )
    if required_columns:
        default_select += ", " + ", ".join(required_columns)
    if select == "":
        select = default_select
    else:
        select = default_select + ", " + select

    # strip any trailing commas and whitespace - these are in the final query
    select = select.rstrip(", ").strip()

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
    input_odb_path: Path,
    output_odb_path: Path,
    sql_query_string: str,
):
    """Subselect ODB data based on an SQL query string using ODC command line tool and
    write to a new ODB file.

    Args:
        input_odb_path (Path): Path to input ODB file.
        output_odb_path (Path): Path to output ODB file.
        sql_query_string (str): SQL query string for ODC.

    Raises:
        FileNotFoundError: If the input ODB file does not exist.
        RuntimeError: If the ODC command in subprocess fails.
    """
    if not Path(input_odb_path).is_file():
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
        print(f"Subsetted ODB written to: {output_odb_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error output: {e.stderr}")
        raise RuntimeError(
            f"ODC SQL command failed with exit code {e.returncode}"
        ) from e


def pivot_obs_df(df: pandas.DataFrame, values: list, columns: list) -> pandas.DataFrame:
    """
    Reshape the dataframe, organised by the values in particular columns.
    For instance, the following dataframe:
    A  B
    1  0.5
    2  0.1
    3  0.6
    1  0.3
    3  0.7
    Using columns=A and values=B, would be reshaped to become:
    B_1  B_2  B_3
    0.5  0.1  0.6
    0.3  NaN  0.7

    Parameters:
        df (pandas.DataFrame): Input DataFrame to pivot.
        columns (list): List of column names - values in these columns will
            be used to define the new columns after the reshaping. For
            instance, channel number or varno.
        values (list): List of column names - values in these columns will
            be spread across different values of the columns. For instance,
            observed value, quality control values.
        Note: all columns not specified in "columns" and "values" will be
            assumed to be "index" values (i.e. are the same within a given
            observation group).
    Returns:
        pandas.DataFrame: Pivoted DataFrame.
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
