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
from unittest.mock import patch

import pytest
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import GetTestData
from anemoi.utils.testing import skip_if_offline

from anemoi.datasets.commands.compare import compare_anemoi_datasets

from .utils.create import create_dataset
from .utils.mock_sources import LoadSource

HERE = os.path.dirname(__file__)
# find_yamls
NAMES = sorted([os.path.basename(path).split(".")[0] for path in glob.glob(os.path.join(HERE, "*.yaml"))])
SKIP = ["recentre"]
SKIP += ["accumulation"]  # test not in s3 yet
SKIP += ["regrid"]
NAMES = [name for name in NAMES if name not in SKIP]
assert NAMES, "No yaml files found in " + HERE


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
        from anemoi.datasets.create.creator import VERSION

        recipe = os.path.join(HERE, name + ".yaml")
        output = os.path.join(HERE, name + ".zarr")

        create_dataset(recipe=recipe, output=output, delta=["12h"])

        directory = get_test_archive(f"anemoi-datasets/create/mock-mars-{VERSION}/{name}.zarr.tgz")
        reference = os.path.join(directory, name + ".zarr")

        errors = compare_anemoi_datasets(reference=reference, actual=output)
        if errors:
            actual_path = os.path.realpath(output)

            print()
            print("⚠️ To update the reference data, run this:")
            print("cd " + os.path.dirname(actual_path))
            base = os.path.basename(actual_path)
            print(f"tar zcf {base}.tgz {base}")
            print(f"scp {base}.tgz data@anemoi.ecmwf.int:public/anemoi-datasets/create/mock-mars-{VERSION}/")
            print()
            raise AssertionError(f"Comparison failed {errors}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Then run pytest
    pytest.main([__file__, "-v", "-k", "nan"])
