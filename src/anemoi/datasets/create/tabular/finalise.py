# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import tqdm

LOG = logging.getLogger(__name__)


def worker(file_path: str):
    LOG.info("✅✅✅✅✅✅✅✅✅ WORKER processing %s", file_path)
    array = np.load(file_path, mmap_mode="r")
    assert len(array.shape) == 2, f"Expected 2D array in {file_path}, got shape {array.shape}"
    duplicates = 0

    # Remove duplicate rows. np.unique does not work well with NaNs, so we replace them with a sentinel value.
    b = np.ascontiguousarray(array)
    b2 = np.nan_to_num(b, nan=np.inf)

    row_dtype = np.dtype((np.void, b2.dtype.itemsize * b2.shape[1]))
    _, idx = np.unique(b2.view(row_dtype), return_index=True)

    unique_array = b[np.sort(idx)]

    LOG.info("✅✅✅✅✅✅✅✅✅ %s %s", array.shape, unique_array.shape)

    if len(unique_array) < len(array):
        duplicates = len(array) - len(unique_array)
        LOG.warning(f"Removed {duplicates} duplicate rows from {file_path}")
        np.save(file_path + ".tmp", unique_array)  # Save to a temporary file first, numpy will add a .npy extension
        os.rename(file_path + ".tmp.npy", file_path)

    first_date = datetime.datetime.fromtimestamp(int(unique_array[0][0]) * 86400 + int(unique_array[0][1]))
    last_date = datetime.datetime.fromtimestamp(int(unique_array[-1][0]) * 86400 + int(unique_array[-1][1]))

    return (first_date, last_date, unique_array.shape, duplicates, file_path)


def finalise_tabular_dataset(work_dir: str, path: str):
    import os

    import numpy as np

    files = [name for name in os.listdir(work_dir) if name.endswith(".npy") and ".tmp" not in name]
    tasks = []
    with ThreadPoolExecutor(max_workers=min(10, os.cpu_count())) as executor:
        for file in files:
            tasks.append(executor.submit(worker, os.path.join(work_dir, file)))

        for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), desc="Checking files", unit="file"):
            print(future.result())

    all_data = []
    for name in files:
        file_path = os.path.join(work_dir, name)
        data = np.load(file_path)
        all_data.append(data)
    final_data = np.concatenate(all_data, axis=0)
    np.save(path, final_data)

    # Clean up temporary files
    for name in files:
        file_path = os.path.join(work_dir, name)
        os.remove(file_path)
