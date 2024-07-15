# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import os

import tqdm
from anemoi.utils.humanize import bytes_to_human

LOG = logging.getLogger(__name__)


def compute_directory_sizes(path):
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
