import logging
import mmap
import os
from multiprocessing import Pool

import eccodes
import numpy as np
import pandas as pd
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


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    force=True,
)


def filter_bufr_message(bid: int, filter_config: dict) -> bool:
    """Check if BUFR message meets filtering conditions specified in filter_config
    Returns True if message should be kept, False if it should be filtered out
    """
    namespace = {"inf": float("inf")}

    for key, condition in filter_config.items():
        try:
            # Get the value from BUFR
            value = eccodes.codes_get(bid, key)

            if isinstance(condition, str) and condition.startswith("lambda"):
                # Lambda expression case
                filter_condition = eval(condition, namespace)
                if not filter_condition(value):
                    return False
            else:

                # Direct value comparison case
                if value != condition:
                    return False

        except eccodes.KeyValueNotFoundError:
            logging.warning(f"Key {key} not found in BUFR message")
            return False
        except Exception as e:
            logging.error(f"Error evaluating condition for {key}: {e}")
            return False

    return True


def bufr_get_array(bid: int, element: str, typ: type, nsubsets: int, missing_val=np.nan) -> np.ndarray:
    """Wrapper for codes_get_array to work around the inconsistent handling of arrays in eccodes when data is constant"""
    try:
        arr = eccodes.codes_get_array(bid, element, typ)
        if len(arr) == 1:
            arr = np.ones(nsubsets, dtype=typ) * arr
    except KeyValueNotFoundError:
        arr = np.ones(nsubsets, dtype=typ) * missing_val
    return arr


def extract_datetimes(bid: int, nreports: int, position_prefix: str = "") -> pd.DataFrame:
    """Extracts and parses the date/time info from a bufr message
    and returns as an array of datetime objects
    """
    df = pd.DataFrame(
        dict(
            years=bufr_get_array(bid, position_prefix + "year", int, nreports),
            months=bufr_get_array(bid, position_prefix + "month", int, nreports),
            days=bufr_get_array(bid, position_prefix + "day", int, nreports),
            hours=bufr_get_array(bid, position_prefix + "hour", int, nreports),
            minutes=bufr_get_array(bid, position_prefix + "minute", int, nreports),
            seconds=bufr_get_array(bid, position_prefix + "second", int, nreports, missing_val=0),
        )
    )
    # Create the datetime series using pandas
    datetimes = pd.to_datetime(df)
    return datetimes


def get_msg(
    bufr_msg,
    per_report: dict,
    prefilter_msg_header: dict = {},
    prefilter_msg_data: dict = {},
    datetime_position_prefix: str = "",
    per_datum: dict = None,
    filters: dict = None,
) -> pd.DataFrame:
    try:
        bid = eccodes.codes_new_from_message(bufr_msg)
        nreports = eccodes.codes_get(bid, "numberOfSubsets")
        eccodes.codes_set(bid, "skipExtraKeyAttributes", 1)

        # Optionally filter messages based on header section entries
        if prefilter_msg_header and not filter_bufr_message(bid, prefilter_msg_header):
            eccodes.codes_release(bid)
            return pd.DataFrame()

        eccodes.codes_set(bid, "unpack", 1)

        # Optionally filter messages based on data section entries
        if prefilter_msg_data and not filter_bufr_message(bid, prefilter_msg_data):
            eccodes.codes_release(bid)
            return pd.DataFrame()

        data_dict = {
            item: bufr_get_array(bid, col, float, nreports).astype(np.float32) for col, item in per_report.items()
        }

        data_dict["times"] = extract_datetimes(bid, nreports, datetime_position_prefix)

        if per_datum:
            for col, sub_dict in per_datum.items():
                ndatum = eccodes.codes_get_size(bid, next(iter(per_datum))) // nreports
                vals = bufr_get_array(bid, col, float, nreports * ndatum).astype(np.float32)
                try:
                    vals_2d = vals.reshape(ndatum, nreports).T
                except ValueError as e:
                    if "cannot reshape array" in str(e):
                        import warnings

                        warnings.warn(
                            f"Reshape error in bufr message {bufr_msg}: Cannot reshape array of size {len(vals)} "
                            f"into shape ({ndatum}, {nreports}). Skipping this message.",
                            RuntimeWarning,
                        )
                        eccodes.codes_release(bid)
                        return None
                    else:
                        raise  # Re-raise if it's a different ValueError

                for col_rename, slice_str in sub_dict.items():
                    vals_col = vals_2d[:, eval(slice_str)]
                    for k in range(vals_col.shape[1]):
                        data_dict[f"{col_rename}_{k+1}"] = vals_col[:, k]

        df = pd.DataFrame(data_dict)

        if filters:
            df = filter_values(df, filters)

        eccodes.codes_release(bid)
        return df
    except Exception as e:
        import warnings

        warnings.warn(
            f"Unexpected error in message: {str(e)}. Skipping this message.",
            RuntimeWarning,
        )
        if "bid" in locals():
            eccodes.codes_release(bid)
        return None


class BufrData(object):
    def __init__(self, BufrFileName):
        self._filename = BufrFileName
        self._fobj = open(self._filename, "rb")
        self._fileno = self._fobj.fileno()
        self._nmsg = eccodes.codes_count_in_file(self._fobj)
        self._dataBlock = self.get_datablock()
        self._lstOffsets = self.get_list_offsets()

    @property
    def dataBlock(self):
        return self._dataBlock

    @property
    def nmsg(self):
        return self._nmsg

    @property
    def lstOffsets(self):
        return self._lstOffsets

    def get_datablock(self):
        with mmap.mmap(self._fileno, length=0, access=mmap.ACCESS_READ) as mobj:
            data = mobj.read()
        return data

    def get_list_offsets(self):
        lstOffsets = []
        for _ in range(0, self._nmsg):
            bid = eccodes.codes_bufr_new_from_file(self._fobj)
            offset = eccodes.codes_get_message_offset(bid)
            size = eccodes.codes_get_message_size(bid)
            lstOffsets.append((offset, size))
            eccodes.codes_release(bid)
        return lstOffsets

    def __del__(self):
        self._fobj.close()


def read_block(
    sublist,
    dataBlock,
    per_report: dict,
    prefilter_msg_header: dict = None,
    prefilter_msg_data: dict = None,
    datetime_position_prefix: str = "",
    per_datum: dict = None,
    filters: dict = None,
):
    log.info(f"PID : {os.getpid()} in read block sublist has {len(sublist)} elements")
    try:
        df_lst = [
            get_msg(
                dataBlock[offset : offset + ch_size],
                per_report,
                prefilter_msg_header,
                prefilter_msg_data,
                datetime_position_prefix,
                per_datum,
                filters,
            )
            for offset, ch_size in sublist
        ]
        return pd.concat(df_lst)
    except Exception as e:
        log.error(f"Error in read_block: {str(e)}")
        raise


def split_list(alist, nparts):
    nelem = len(alist)
    chunkSize = nelem // (nparts)
    sublists = []
    for i in range(0, nelem, chunkSize):
        slist = alist[i : i + chunkSize]
        sublists.append(slist)
    return sublists


def bufr2df_parallel(
    ekd_ds: BUFRReader,
    per_report: dict,
    nproc: int = 1,
    prefilter_msg_header: dict = None,
    prefilter_msg_data: dict = None,
    datetime_position_prefix: str = "",
    per_datum: dict = None,
    filters: dict = None,
) -> pd.DataFrame:
    fname = ekd_ds.path
    mbfo = BufrData(fname)
    fullDataBlock = mbfo.dataBlock
    log.info(f"number of messages {mbfo.nmsg}")
    sublists = split_list(mbfo.lstOffsets, nproc)

    nSubLists = len(sublists)

    pool = Pool(processes=nproc)
    try:
        results = [
            pool.apply_async(
                read_block,
                args=(
                    sublists[i],
                    fullDataBlock,
                    per_report,
                    prefilter_msg_header,
                    prefilter_msg_data,
                    datetime_position_prefix,
                    per_datum,
                    filters,
                ),
            )
            for i in range(0, nSubLists)
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
        df = df.sort_values(by=["times"]).reset_index(drop=True)

    log.info(f"Number of rows in the dataframe {len(df)}")

    return df
