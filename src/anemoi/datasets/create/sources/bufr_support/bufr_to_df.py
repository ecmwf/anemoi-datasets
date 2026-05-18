# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import operator
import os
import warnings
from collections.abc import Sequence
from itertools import batched
from multiprocessing import Pool
from typing import Any
from typing import Literal

import eccodes
import numpy as np
import pandas as pd
from pandas import DataFrame

from .bufr_reader import BUFRMessage
from .bufr_reader import BUFRReader

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    force=True,
)


class BUFRMessageSelector:
    """For evaluating whether a BUFR message matches a given condition.

    Args:
        key: The BUFR key name to evaluate in the message (e.g. ``"dataCategory"``).
        condition: A two-element list ``[operator, value]`` describing the selection.
            The operator can be a comparison string such as ``"=="``, ``"!="``,
            ``">"``, ``">="``, ``"<"``, ``"<="``, ``"in"``, or ``"not in"``.
            The value is the expected reference value to compare against.  For
            ``"in"`` / ``"not in"`` operators the value must be a list.  The
            special string ``"inf"`` is converted to ``float("inf")``.
    """

    _OPERATOR_ALIASES = {
        "==": "eq",
        "!=": "ne",
        ">": "gt",
        ">=": "ge",
        "<": "lt",
        "<=": "le",
    }

    def __init__(self, key: str, condition: Any):
        self._check_inputs(key, condition)

        self.key = key
        self.operator, self.expected = condition
        if isinstance(self.expected, str) and self.expected == "inf":
            self.expected = float("inf")
        self.operator = self._normalise_operator(self.operator)
        self._operation = self._get_operator(self.operator)

    @classmethod
    def _get_operator(cls, op: str):
        op = cls._OPERATOR_ALIASES.get(op, op)
        if op == "in":
            return cls._in
        elif op == "not in":
            return cls._not_in
        try:
            return getattr(operator, op)
        except AttributeError as e:
            raise ValueError(f"Invalid operator: {op}") from e

    @staticmethod
    def _normalise_operator(operator_str: str) -> str:
        return operator_str.strip().lower()

    @staticmethod
    def _in(a, b):
        return a in b

    @staticmethod
    def _not_in(a, b):
        return a not in b

    @classmethod
    def _check_inputs(cls, key: str, condition: Any):
        if not isinstance(condition, list) or len(condition) != 2:
            raise ValueError(
                f"Invalid selection for '{key}': expected a list of length 2 [operator, value], got {condition!r}."
            )

        op, expected = condition
        if not isinstance(op, str):
            raise ValueError(f"Invalid selection for '{key}': operator must be a string, got {type(op)}.")

        op = cls._normalise_operator(op)
        if op in {"in", "not in"} and not isinstance(expected, list):
            raise ValueError(
                f"Invalid selection for '{key}': operator '{op}' requires a list value, got {type(expected)}."
            )

    def matches(self, msg: BUFRMessage) -> bool:
        value = msg.get_value(self.key)
        return self._operation(value, self.expected)


def parse_slice(s: str) -> slice:
    """Convert a string like ``"1:5"`` or ``"::2"`` into a Python :class:`slice`.

    Accepted formats mirror Python slice syntax: ``start:stop`` or
    ``start:stop:step``.  Each component may be omitted (treated as
    ``None``).  A bare integer string (e.g. ``"3"``) is interpreted as
    ``slice(3, 4)`` so that it selects a single element while keeping
    the result as a slice.

    Raises:
        ValueError: If the string cannot be parsed as a valid slice.
    """
    if not isinstance(s, str):
        raise ValueError(f"Expected a string, got {type(s).__name__}")

    s = s.strip()
    if not s:
        raise ValueError("Empty string cannot be converted to a slice")

    parts = s.split(":")

    if len(parts) == 1:
        # integer to single-element slice
        idx = _parse_slice_component(parts[0])
        if idx is None:
            raise ValueError(f"Invalid slice string: {s!r}")
        stop = None if idx == -1 else idx + 1
        return slice(idx, stop)
    elif len(parts) in (2, 3):
        components = [_parse_slice_component(p) for p in parts]
        return slice(*components)
    else:
        raise ValueError(f"Invalid slice string: {s!r}")


def _parse_slice_component(component: str) -> int | None:
    component = component.strip()
    if not component:
        return None
    try:
        return int(component)
    except ValueError:
        raise ValueError(f"Invalid slice component: {component!r}")


class BUFRToDataFrame:
    """Convert BUFR messages into pandas DataFrames.

    Args:
        per_report: Mapping of BUFR key names to output column names for
            values that occur once per observation report (subset).  Each key
            is extracted as a 1-D array of length ``numberOfSubsets`` (as found
            in the BUFR message header).
        preselect_msg_header: Optional dictionary of header-section selection
            conditions applied _before_ the message is unpacked.  Each entry
            maps a BUFR key to a ``[operator, value]`` pair (see
            :class:`BUFRMessageSelector`).  Messages that do not satisfy all
            conditions are skipped.
        preselect_msg_data: Optional dictionary of data-section selection
            conditions applied -after_ the message is unpacked.  Format is
            the same as ``preselect_msg_header``.
        datetime_position_prefix: Prefix prepended to the standard date/time
            key names (``year``, ``month``, ``day``, ``hour``, ``minute``,
            ``second``) when extracting observation timestamps.  Defaults to
            an empty string.
        per_datum: Optional mapping describing variables that have multiple
            values per report. The expected structure depends on
            ``per_datum_format``:

            * ``"wide"`` (default) – ``{bufr_key: {col_name: slice_str, ...}, ...}``
              where ``slice_str`` is a Python slice expression applied to the
              reshaped 2-D array.
            * ``"long"`` – ``{bufr_key: {"name": col_name, "start_index": int}, ...}``
              where each BUFR key is unpacked into a single column and the
              per-report rows are repeated accordingly.
        per_datum_format: Either ``"wide"`` or ``"long"``.  Controls how
            ``per_datum`` columns are arranged in the resulting DataFrame.
            Defaults to ``"wide"``.
    """

    def __init__(
        self,
        *,
        per_report: dict,
        preselect_msg_header: dict = None,
        preselect_msg_data: dict = None,
        datetime_position_prefix: str = "",
        per_datum: dict = None,
        per_datum_format: Literal["long", "wide"] = "wide",
    ):
        self.per_report = per_report
        self.preselect_msg_header = self._build_selectors(preselect_msg_header)
        self.preselect_msg_data = self._build_selectors(preselect_msg_data)
        self.datetime_position_prefix = datetime_position_prefix

        self.per_datum = per_datum or {}

        if per_datum_format not in ("long", "wide"):
            raise ValueError(f"Invalid per_datum_format: '{per_datum_format}'. Must be 'wide' or 'long'.")

        if self.per_datum:
            if per_datum_format == "long":
                self.per_datum_processing = self._get_msg_long_df
            elif per_datum_format == "wide":
                self.per_datum_processing = self._get_msg_wide_df

    @staticmethod
    def _build_selectors(conditions: dict[str, Any] | None) -> list[BUFRMessageSelector]:
        if conditions is None:
            return []
        if not isinstance(conditions, dict):
            raise ValueError(
                "Invalid BUFR preselect config: expected a dictionary of key: [operator, value] conditions."
            )
        return [BUFRMessageSelector(key, condition) for key, condition in conditions.items()]

    @staticmethod
    def filter_bufr_message(message: BUFRMessage, selectors: list[BUFRMessageSelector]) -> bool:
        """Check if the BUFR message meets selection conditions specified in config
        Returns False if the message should be kept, True if it should be filtered (discarded).
        """
        if not selectors:
            return False

        for message_selector in selectors:
            try:
                if not message_selector.matches(message):
                    return True
            except eccodes.KeyValueNotFoundError:
                logging.warning(f"Key {message_selector.key} not found in BUFR message")
                return True
            except Exception as e:
                logging.error(f"Error evaluating condition for {message_selector.key}: {e}")
                return True
        return False

    def extract_datetimes(self, message: BUFRMessage, nreports: int) -> np.ndarray:
        """Extracts and parses the date/time info from a bufr message
        and returns as an array of datetime objects
        """
        df = pd.DataFrame(
            dict(
                years=message.get_array(self.datetime_position_prefix + "year", int, nreports),
                months=message.get_array(self.datetime_position_prefix + "month", int, nreports),
                days=message.get_array(self.datetime_position_prefix + "day", int, nreports),
                hours=message.get_array(self.datetime_position_prefix + "hour", int, nreports),
                minutes=message.get_array(self.datetime_position_prefix + "minute", int, nreports),
                seconds=message.get_array(self.datetime_position_prefix + "second", int, nreports, missing_val=0),
            )
        )
        # Create the datetime series using pandas
        return pd.to_datetime(df).values

    def read_msg_to_df(self, message: BUFRMessage) -> pd.DataFrame:
        try:
            nreports = message.get_value("numberOfSubsets")
            message.set_value("skipExtraKeyAttributes", 1)

            # Optionally filter messages based on header section entries
            if self.filter_bufr_message(message, self.preselect_msg_header):
                return pd.DataFrame()

            message.set_value("unpack", 1)

            # Optionally filter messages based on data section entries
            if self.filter_bufr_message(message, self.preselect_msg_data):
                return pd.DataFrame()

            df = self.extract_per_report_df(message, nreports)

            if self.per_datum:
                df = self.per_datum_processing(message, df, nreports)

            return df
        except Exception as e:
            warnings.warn(
                f"Unexpected error in message: {str(e)}. Skipping this message.",
                RuntimeWarning,
            )
            return pd.DataFrame()

    def extract_per_report_df(self, message: BUFRMessage, nreports: str | None | Any) -> DataFrame:
        per_report_data = {
            item: message.get_array(col, float, nreports).astype(np.float32) for col, item in self.per_report.items()
        }
        per_report_data["datetime"] = self.extract_datetimes(message, nreports)
        df = pd.DataFrame(per_report_data)
        return df

    def _get_msg_long_df(self, message: BUFRMessage, df_report: DataFrame, nreports: int) -> DataFrame:
        # Extract the per-datum arrays (now just 1D arrays)
        df_datum = pd.DataFrame()

        # For 'long' format, per_datum is expected to be a simple {bufr_key: new_column_name} dict
        for col, item_config in self.per_datum.items():
            new_name = item_config["name"]
            start_index = int(item_config.get("start_index", 0))

            ndatum = message.get_size(col) // nreports
            ndatum = len(np.ones(ndatum)[start_index:])
            vals = message.get_array(col, float, ndatum).astype(np.float32)

            vals = vals[start_index:]

            df_datum[new_name] = vals

        # repeat per-report data for each datum
        df_expanded_reports = df_report.loc[df_report.index.repeat(ndatum)].reset_index(drop=True)

        # Combine the expanded report data and the datum data
        return pd.concat([df_expanded_reports, df_datum], axis=1)

    def _get_msg_wide_df(self, message: BUFRMessage, df_report: DataFrame, nreports: int) -> DataFrame:
        # Logic for 'wide' format (column-wise) ---
        data_dict = df_report.to_dict("list")  # Start with the per-report data

        for col, sub_dict in self.per_datum.items():
            # Note: This assumes a fixed number of levels across reports in a message
            ndatum = message.get_size(col) // nreports
            vals = message.get_array(col, float, nreports * ndatum).astype(np.float32)
            try:
                vals_2d = vals.reshape(nreports, ndatum)
            except ValueError as e:
                if "cannot reshape array" in str(e):
                    warnings.warn(
                        f"Reshape error in bufr message: Cannot reshape array of size {len(vals)} "
                        f"into shape ({nreports}, {ndatum}). Skipping this message.",
                        RuntimeWarning,
                    )
                    return pd.DataFrame()
                raise

            for col_rename, slice_op in sub_dict.items():
                if isinstance(slice_op, str):
                    slice_op = parse_slice(slice_op)
                vals_col = vals_2d[:, slice_op]
                # Handle cases where slice results in a 1D or 2D array
                if vals_col.ndim == 1:
                    data_dict[f"{col_rename}_1"] = vals_col
                else:
                    for k in range(vals_col.shape[1]):
                        data_dict[f"{col_rename}_{k + 1}"] = vals_col[:, k]
        return pd.DataFrame(data_dict)

    def read_bufr_to_df(
        self, bufr_reader: "BUFRReader", msg_idxs: Sequence[int] | None = None, sort=True
    ) -> pd.DataFrame:
        if msg_idxs is None:
            msg_idxs = range(len(bufr_reader))

        log.info(f"PID : {os.getpid()} in read_bufr_to_df has {len(msg_idxs)} elements")
        try:
            df_lst = []
            for msg_idx in msg_idxs:
                with bufr_reader.get_message(msg_idx) as message:
                    df_lst.append(self.read_msg_to_df(message))
            df = pd.concat(df_lst)
        except Exception as e:
            log.error(f"Error in read_bufr_to_df: {str(e)}")
            raise

        if len(df) > 0 and sort:
            df = df.sort_values(by=["datetime"]).reset_index(drop=True)
        return df


def bufr_to_dataframe_parallel(bufr_reader: BUFRReader, bufr_to_df: BUFRToDataFrame, nproc: int = 1) -> pd.DataFrame:
    def split_list(alist, nparts):
        chunk_size = max(1, len(alist) // nparts)
        return [list(chunk) for chunk in batched(alist, chunk_size)]

    num_msgs = len(bufr_reader)
    log.info(f"Number of messages {num_msgs}")
    sublists_idxs = split_list(range(num_msgs), nproc)

    pool = Pool(processes=nproc)
    try:
        results = [
            pool.apply_async(
                bufr_to_df.read_bufr_to_df,
                args=(
                    bufr_reader,
                    idxs,
                ),
                # sort = False because we sort after combining
                kwds={"sort": False},
            )
            for idxs in sublists_idxs
        ]
        all_lst = []
        for r in results:
            try:
                df = r.get()
                all_lst.append(df)
            except Exception as e:
                log.error(f"Error getting result from worker process: {str(e)}")
                continue
        if not all_lst:
            raise ValueError("No valid results were returned from any worker process")
    finally:
        pool.close()  # Stop accepting new tasks
        pool.join()  # Wait for workers to finish with timeout
        pool.terminate()  # Force terminate if still running

    df = pd.concat(all_lst)
    if len(df) > 0:
        df = df.sort_values(by=["datetime"]).reset_index(drop=True)

    log.info(f"Number of rows in the dataframe {len(df)}")
    return df
