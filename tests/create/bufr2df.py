import eccodes
import numpy as np
import pandas as pd
import tqdm
from earthkit.data.readers.bufr.bufr import BUFRReader
from gribapi.errors import KeyValueNotFoundError


def filter_values(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter the DataFrame based on the specified conditions"""
    for col, condition in filters.items():
        if isinstance(condition, str):
            condition = eval(condition)
        if callable(condition):
            df = df[df[col].apply(condition)]
        elif isinstance(condition, slice):
            start, stop = condition.start, condition.stop
            query_str = f"({start} <= {col}) & ({col} < {stop})"
            df = df.query(query_str)
        elif isinstance(condition, (list, set)):
            df = df[df[col].isin(condition)]
        else:
            raise ValueError(f"Invalid condition for column '{col}': {condition}")
    return df


def bufr_get_array(bid: int, element: str, typ: type, nsubsets: int, missing_val=np.nan) -> np.ndarray:
    """Wrapper for codes_get_array to work around the inconsistent handling of arrays in eccodes when data is constant"""
    try:
        arr = eccodes.codes_get_array(bid, element, typ)
        if len(arr) == 1:
            arr = np.ones(nsubsets, dtype=typ) * arr
    except KeyValueNotFoundError:
        arr = np.ones(nsubsets, dtype=typ) * missing_val
    return arr


def extract_datetimes(bid: int, nreports: int) -> pd.DataFrame:
    """Extracts and parses the date/time info from a bufr message
    and returns as an array of datetime objects
    """
    df = pd.DataFrame(
        dict(
            years=bufr_get_array(bid, "year", int, nreports),
            months=bufr_get_array(bid, "month", int, nreports),
            days=bufr_get_array(bid, "day", int, nreports),
            hours=bufr_get_array(bid, "hour", int, nreports),
            minutes=bufr_get_array(bid, "minute", int, nreports),
            seconds=bufr_get_array(bid, "second", int, nreports, missing_val=0),
        )
    )
    # Create the datetime series using pandas
    datetimes = pd.to_datetime(df)
    return datetimes


def get_msg(f, i, per_report_dict, per_datum_dict=None, filters=None) -> pd.DataFrame:
    bid = eccodes.codes_bufr_new_from_file(f)
    eccodes.codes_set(bid, "unpack", 1)
    nreports = eccodes.codes_get(bid, "numberOfSubsets")

    data_dict = {
        item: bufr_get_array(bid, col, float, nreports).astype(np.float32) for col, item in per_report_dict.items()
    }
    data_dict["times"] = extract_datetimes(bid, nreports)

    if per_datum_dict:
        for col, sub_dict in per_datum_dict.items():
            ndatum = eccodes.codes_get_size(bid, next(iter(per_datum_dict))) // nreports
            vals = bufr_get_array(bid, col, float, nreports * ndatum).astype(np.float32)
            try:
                vals_2d = vals.reshape(ndatum, nreports).T
            except ValueError as e:
                if "cannot reshape array" in str(e):
                    import warnings

                    warnings.warn(
                        f"Reshape error in file {f}, message {i}: Cannot reshape array of size {len(vals)} "
                        f"into shape ({ndatum}, {nreports}). Skipping this message.",
                        RuntimeWarning,
                    )
                    eccodes.codes_release(bid)
                    return None
                else:
                    raise  # Re-raise if it's a different ValueError

            for col_rename, slice_str in sub_dict.items():
                vals_col = vals_2d[:, eval(slice_str)]
                for i in range(vals_col.shape[1]):
                    data_dict[f"{col_rename}_{i+1}"] = vals_col[:, i]

    df = pd.DataFrame(data_dict)

    if filters:
        df = filter_values(df, filters)

    eccodes.codes_release(bid)
    return df


def bufr2df(
    ekd_ds: BUFRReader,
    per_report: dict,
    per_datum: dict = None,
    filter: dict = None,
) -> pd.DataFrame:
    """Extracts data from a BUFR file into a pandas DataFrame
    -info on what to extract (and how it should be named in the dataframe) are
     provided by input dictionaries; one at the per-report level and another for the per-datum
    """
    fname = ekd_ds.path
    with open(fname, "rb") as f:
        nmessages = eccodes.codes_count_in_file(f)
        bar = tqdm.tqdm(
            iterable=range(nmessages),
            desc="Processing bufr messages...",
            mininterval=20.0,
        )
        df_lst = [get_msg(f, i, per_report, per_datum, filter) for i in bar]
    df = pd.concat(df_lst)
    df = df.sort_values(by=["times"]).reset_index(drop=True)
    return df
