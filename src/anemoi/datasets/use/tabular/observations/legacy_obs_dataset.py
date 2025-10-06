# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)


class ObsDataset(Dataset):

    def __init__(
        self,
        filename: str,
        start: int,
        end: int,
        len_hrs: int,
        step_hrs: int = None,
        select: list[str] = None,
        drop: list[str] = None,
    ) -> None:

        self.filename = filename
        self.z = zarr.open(filename, mode="r")
        self.data = self.z["data"]
        self.dt = self.z["dates"]  # datetime only
        self.hrly_index = self.z["idx_197001010000_1"]
        self.colnames = self.data.attrs["colnames"]
        self.selected_colnames = self.colnames
        self.selected_cols_idx = np.arange(len(self.colnames))
        self.len_hrs = len_hrs
        self.step_hrs = step_hrs if step_hrs else len_hrs

        # Create index for samples
        self._setup_sample_index(start, end, self.len_hrs, self.step_hrs)

        self._load_properties()

        if select:
            self.select(select)

        if drop:
            self.drop(drop)

    def __getitem__(
        self,
        idx: int,
    ) -> torch.tensor:

        start_row = self.indices_start[idx]
        end_row = self.indices_end[idx]

        data = self.data.oindex[start_row:end_row, self.selected_cols_idx]

        return torch.from_numpy(data)

    def __len__(self) -> int:

        return len(self.indices_start)

    def get_dates(
        self,
        idx: int,
    ) -> np.ndarray:

        start_row = self.indices_start[idx]
        end_row = self.indices_end[idx]
        dates = self.dt.oindex[start_row:end_row]

        assert len(dates.shape) == 2, dates.shape
        dates = dates[:, 0]

        if len(dates) and dates[0].dtype != np.dtype("datetime64[s]"):
            dates = dates.astype("datetime64[s]")

        return dates

    def get_df(self, idx: int) -> pd.DataFrame:
        """Convenience function to return data for sample idx packaged in a pandas DataFrame"""

        d = self.__getitem__(idx)

        df = pd.DataFrame(data=d, columns=[self.colnames[i] for i in self.selected_cols_idx])

        start_row = self.indices_start[idx]
        end_row = self.indices_end[idx]

        dts = self.dt[start_row:end_row, :]
        df["datetime"] = dts

        return df

    def select(self, cols_list: list[str]) -> None:
        """Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.
        """
        self.selected_colnames = cols_list
        self.selected_cols_idx = np.array([self.colnames.index(item) for item in cols_list])

    def drop(self, cols_to_drop: list[str]) -> None:
        """Allow user to drop specific columns from the dataset.
        Get functions no longer return data for these columns after being set.
        """
        mask = [name not in cols_to_drop for name in self.selected_colnames]

        self.selected_colnames = [name for name, keep in zip(self.selected_colnames, mask) if keep]
        self.selected_cols_idx = self.selected_cols_idx[mask]

    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        """Returns a tuple of datetime objects describing the start and end times of the sample at position idx."""

        if idx < 0:
            idx = len(self) + idx

        time_start = self.start_dt + datetime.timedelta(hours=(idx * self.step_hrs), seconds=1)
        time_end = min(
            self.start_dt + datetime.timedelta(hours=(idx * self.step_hrs + self.len_hrs)),
            self.end_dt,
        )

        return (np.datetime64(time_start), np.datetime64(time_end))

    def first_sample_with_data(self) -> int:
        """Returns the position of the first sample which contains data."""
        return int(np.nonzero(self.indices_end)[0][0]) if self.indices_end.max() > 0 else None

    def last_sample_with_data(self) -> int:
        """Returns the position of the last sample which contains data."""
        if self.indices_end.max() == 0:
            last_sample = None
        else:
            last_sample = int(np.where(np.diff(np.append(self.indices_end, self.indices_end[-1])) > 0)[0][-1] + 1)

        return last_sample

    def _setup_sample_index(self, start: int, end: int, len_hrs: int, step_hrs: int) -> None:
        """Dataset is divided into samples;
        - each n_hours long
        - sample 0 starts at start (yyyymmddhhmm)
        - index array has one entry for each sample; contains the index of the first row
        containing data for that sample
        """

        try:
            from obsdata.config import config

            assert config.base_index_yyyymmddhhmm == 197001010000, "base_index_yyyymmddhhmm must be 197001010000"
        except ImportError:
            pass
        base_yyyymmddhhmm = 197001010000

        assert start > base_yyyymmddhhmm, (
            f"Abort: ObsDataset sample start (yyyymmddhhmm) must be greater than {base_yyyymmddhhmm}\n"
            f"       Current value: {start}"
        )

        format_str = "%Y%m%d%H%M%S"
        base_dt = datetime.datetime.strptime(str(base_yyyymmddhhmm), format_str)
        self.start_dt = datetime.datetime.strptime(str(start), format_str)
        self.end_dt = datetime.datetime.strptime(str(end), format_str)

        # Calculate hours since the base date for the requested dataset ranges
        diff_in_hours_start = int((self.start_dt - base_dt).total_seconds() // 3600)
        diff_in_hours_end = int((self.end_dt - base_dt).total_seconds() // 3600)

        # Find elements that need to be extracted from the hourly index
        #  + ensuring that the dataset respects the requested end-hour even if it is mid-way through a sample
        sample_starts = np.arange(diff_in_hours_start, diff_in_hours_end, step_hrs)
        sample_ends = np.minimum(sample_starts + len_hrs, diff_in_hours_end)

        # Initialize local index arrays
        self.indices_start = np.zeros(sample_starts.shape, dtype=int)
        self.indices_end = np.zeros(self.indices_start.shape, dtype=int)

        max_hrly_index = self.hrly_index.shape[0] - 1
        valid_start_mask = sample_starts <= max_hrly_index
        valid_end_mask = (sample_ends > 0) & (sample_ends <= max_hrly_index)

        # Copy elements from the hrly_index into the local index
        self.indices_start[valid_start_mask] = self.hrly_index[sample_starts[valid_start_mask]]
        self.indices_end[valid_end_mask] = np.maximum(self.hrly_index[sample_ends[valid_end_mask]], 0)

    def _load_properties(self) -> None:

        self.properties = {}

        self.properties["means"] = self.data.attrs["means"]
        self.properties["vars"] = self.data.attrs["vars"]
        self.properties["data_idxs"] = self.data.attrs["data_idxs"]
        self.properties["obs_id"] = self.data.attrs["obs_id"]
