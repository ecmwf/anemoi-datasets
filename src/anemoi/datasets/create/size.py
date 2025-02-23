# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from typing import Dict
from typing import Optional

import tqdm
from anemoi.utils.humanize import bytes_to_human

LOG = logging.getLogger(__name__)


def compute_directory_sizes(path: str) -> Optional[Dict[str, int]]:
    """Computes the total size and number of files in a directory.

    Parameters
    ----------
    path : str
        The path to the directory.

    Returns
    -------
    dict of str to int or None
        A dictionary with the total size and number of files, or None if the path is not a directory.
    """
    if not os.path.isdir(path):
        return None

    size, n = 0, 0
    bar = tqdm.tqdm(iterable=os.walk(path), desc=f"Computing size of {path}")
    for dirpath, _, filenames in bar:
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            size += os.path.getsize(file_path)
            n += 1

    LOG.info(f"Total size: {bytes_to_human(size)}")
    LOG.info(f"Total number of files: {n}")

    return dict(total_size=size, total_number_of_files=n)
