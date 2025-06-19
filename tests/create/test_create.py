# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import glob
import logging
import os
import sys

import pytest
from anemoi.utils.testing import get_test_archive
from anemoi.utils.testing import skip_if_offline

from .utils.compare import Comparer
from .utils.create import create_dataset
from .utils.mock_sources import mockup_from_source

HERE = os.path.dirname(__file__)
# find_yamls
NAMES = sorted([os.path.basename(path).split(".")[0] for path in glob.glob(os.path.join(HERE, "*.yaml"))])
SKIP = ["recentre"]
SKIP += ["accumulation"]  # test not in s3 yet
SKIP += ["regrid"]
NAMES = [name for name in NAMES if name not in SKIP]
assert NAMES, "No yaml files found in " + HERE


@skip_if_offline
@pytest.mark.parametrize("name", NAMES)
@mockup_from_source
def test_run(name: str) -> None:
    """Run the test for the specified dataset.

    Parameters
    ----------
    name : str
        The name of the dataset.

    Raises
    ------
    AssertionError
        If the comparison fails.
    """
    config = os.path.join(HERE, name + ".yaml")
    output = os.path.join(HERE, name + ".zarr")
    is_test = False

    create_dataset(config=config, output=output, delta=["12h"], is_test=is_test)

    directory = get_test_archive(f"anemoi-datasets/create/mock-mars/{name}.zarr.tgz")
    reference = os.path.join(directory, name + ".zarr")

    Comparer(output_path=output, reference_path=reference).compare()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = NAMES

    for name in names:
        logging.info(f"Running test for {name}")
        try:
            test_run(name)
        except AssertionError:
            pass
