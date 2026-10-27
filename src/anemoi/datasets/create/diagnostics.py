# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Diagnostics that sources record while a dataset is built (e.g. QC counts), kept with it.

A source calls ``context.add_diagnostics(name, value)``. ``load`` pickles each group's to
``<work_dir>/diagnostics/``; ``finalise`` merges them into ``<dataset>/_monitor/data/``:

- a DataFrame per name: concatenated, ``<name>.parquet``
- a numpy array per name: summed per month of the group's first date, ``<name>.npz`` with
  one key per ``YYYY-MM``

``anemoi-datasets monitor`` plots them.
"""

import glob
import logging
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

FOLDER = "_monitor"


def save(work_dir: str, group: int, first_date, diagnostics: dict) -> None:
    """Pickle one group's diagnostics (atomic: a part killed mid-write leaves no file)."""
    folder = os.path.join(work_dir, "diagnostics")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{group:06d}.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump({"group": group, "first_date": pd.Timestamp(first_date), "diagnostics": diagnostics}, f)
    os.replace(path + ".tmp", path)


def merge(work_dir: str, path: str, n_groups: int) -> list[str]:
    """Merge every group's diagnostics into <path>/_monitor/data/; returns the files written.

    Raises if some groups recorded diagnostics and others did not.
    """
    files = sorted(glob.glob(os.path.join(work_dir, "diagnostics", "*.pkl")))
    if not files:
        return []
    if len(files) != n_groups:
        raise ValueError(f"diagnostics: {len(files)} of {n_groups} groups have diagnostics in {work_dir}")
    frames, arrays = defaultdict(list), defaultdict(dict)
    with ThreadPoolExecutor(32) as pool:  # thousands of small files: bound by file system latency
        entries = list(pool.map(_load, files))
    for entry in entries:
        month = entry["first_date"].strftime("%Y-%m")
        for name, value in entry["diagnostics"].items():
            if isinstance(value, pd.DataFrame):
                frames[name].append(value)
            elif isinstance(value, np.ndarray):
                arrays[name][month] = arrays[name][month] + value if month in arrays[name] else value.copy()
            else:
                raise TypeError(f"diagnostics {name!r}: a DataFrame or a numpy array, not {type(value).__name__}")
    out = os.path.join(path, FOLDER, "data")
    os.makedirs(out, exist_ok=True)
    written = []
    for name, parts in frames.items():
        written.append(os.path.join(out, f"{name}.parquet"))
        pd.concat(parts, ignore_index=True).to_parquet(written[-1], index=False)
    for name, months in arrays.items():
        written.append(os.path.join(out, f"{name}.npz"))
        np.savez(written[-1], **months)
    LOG.info(f"Merged the diagnostics of {len(files)} groups into {out}")
    return written


def _load(file: str) -> dict:
    with open(file, "rb") as f:
        return pickle.load(f)
