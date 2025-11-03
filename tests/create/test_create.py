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
from unittest.mock import patch

import pytest
import yaml
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import GetTestData
from anemoi.utils.testing import skip_if_offline

from .utils.checks import check_dataset
from .utils.create import create_dataset
from .utils.mock_sources import LoadSource

HERE = os.path.dirname(__file__)
# find_yamls

NAMES = []
for path in glob.glob(os.path.join(HERE, "*.yaml")):
    name, _ = os.path.splitext(os.path.basename(path))
    with open(path) as f:
        conf = yaml.safe_load(f)
        if conf.get("skip_test", False):
            continue
        if conf.get("slow_test", False):
            NAMES.append(pytest.param(name, marks=pytest.mark.slow))
            continue
    NAMES.append(name)


# Used by pipe.yaml
@filter_registry.register("filter")
class TestFilter(Filter):

    def __init__(self, **kwargs):

        self.kwargs = kwargs

    def forward(self, data):
        return data.sel(**self.kwargs)


@pytest.fixture
def load_source(get_test_data: GetTestData) -> LoadSource:
    return LoadSource(get_test_data)


@skip_if_offline
@pytest.mark.parametrize("name", NAMES)
def test_run(name: str, get_test_archive: GetTestArchive, load_source: LoadSource) -> None:
    """Run the test for the specified dataset.

    Parameters
    ----------
    name : str
        The name of the dataset.
    get_test_archive : callable
        Fixture to retrieve the test archive.
    load_source : LoadSource
        Fixture to mock data sources.

    Raises
    ------
    AssertionError
        If the comparison fails.
    """
    with patch("earthkit.data.from_source", load_source):
        config = os.path.join(HERE, name + ".yaml")
        output = os.path.join(HERE, name + ".zarr")

        create_dataset(config=config, output=output, delta=["12h"])

        check_dataset(name, config, output, get_test_archive)


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
