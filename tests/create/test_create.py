# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import glob
import json
import logging
import os
from unittest.mock import patch

import pytest
import yaml
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import GetTestData
from anemoi.utils.testing import skip_if_offline

from .utils.create import create_dataset
from .utils.mock_sources import LoadSource

# Don't import any anemoi.datasets* here,
# otherwise the "Filter" filter will be registered too late
# and not be part of the Recipe pydantic model


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
class FilterForTesting(Filter):

    def __init__(self, **kwargs):

        self.kwargs = kwargs

    def forward(self, data):
        return data.sel(**self.kwargs)


@pytest.fixture
def load_source(get_test_data: GetTestData) -> LoadSource:
    return LoadSource(get_test_data)



MARS_REQUESTS = os.path.join(HERE, "requests")

#: Set to 1 to rewrite the recorded requests instead of checking them.
UPDATE_MARS_REQUESTS = os.environ.get("ANEMOI_UPDATE_MARS_REQUESTS") == "1"


def _tally(requests: list) -> dict:
    """Collapse recorded requests to ``md5 -> {count, request}``.

    Counting rather than de-duplicating on purpose: retrieving the same field
    twice is a real defect, not a detail, so it has to show up in the diff.
    """
    tally: dict = {}
    for entry in requests:
        seen = tally.setdefault(entry["md5"], {"count": 0, "request": entry["request"]})
        seen["count"] += 1
    return tally


def _check_mars_requests(name: str, requests: list) -> None:
    """Assert the recipe asked MARS for exactly what it asked for last time.

    The mock fixtures are keyed by an md5 of the request, so any change to which
    fields a recipe retrieves makes the lookup miss -- and ``LoadSource.get_data``
    then calls ``exit(1)``, killing the whole run with nothing useful said. Checking
    the requests first turns that into a readable diff.

    This runs *before* the reference-dataset comparison below, deliberately: "did we
    ask the archive for the right things" is the more fundamental question, and it
    keeps working while a reference is stale.

    Parameters
    ----------
    name : str
        The recipe name; names the file the requests are recorded in.
    requests : list
        What ``LoadSource`` recorded during the build.

    Raises
    ------
    AssertionError
        If the requests differ from the recorded ones.
    """
    path = os.path.join(MARS_REQUESTS, name + ".json")
    tally = _tally(requests)

    if UPDATE_MARS_REQUESTS:
        if tally:
            os.makedirs(MARS_REQUESTS, exist_ok=True)
            with open(path, "w") as f:
                json.dump([dict(md5=k, **v) for k, v in sorted(tally.items())], f, indent=2)
                f.write("\n")
        elif os.path.exists(path):
            os.remove(path)
        return

    if not os.path.exists(path):
        assert not tally, (
            f"{name} issued {len(tally)} MARS request(s) but has no recorded baseline.\n"
            "Regenerate with ANEMOI_UPDATE_MARS_REQUESTS=1"
        )
        return

    with open(path) as f:
        expected = {e["md5"]: {"count": e["count"], "request": e["request"]} for e in json.load(f)}

    errors = []
    for md5 in sorted(set(expected) | set(tally)):
        was, now = expected.get(md5), tally.get(md5)
        if was is None:
            errors.append(f"  + asked for something new: {json.dumps(now['request'])}")
        elif now is None:
            errors.append(f"  - no longer asked for:     {json.dumps(was['request'])}")
        elif was["count"] != now["count"]:
            errors.append(
                f"  ~ retrieved {was['count']}x -> {now['count']}x: {json.dumps(now['request'])}"
            )

    assert not errors, (
        f"{name} no longer asks MARS for the same fields:\n"
        + "\n".join(errors)
        + "\n\nIf the change is intended, regenerate with ANEMOI_UPDATE_MARS_REQUESTS=1"
    )


SKIPPED_TESTS = ["recentre"]


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
    if name in SKIPPED_TESTS:
        pytest.skip("Not ready yet")

    import requests

    with (
        patch("earthkit.data.from_source", load_source),
        patch("anemoi.datasets.create.sources.mars.retrieval.from_source", load_source),
    ):
        from anemoi.datasets.create.creator import VERSION

        recipe = os.path.join(HERE, name + ".yaml")
        output = os.path.join(HERE, name + ".zarr")

        create_dataset(recipe=recipe, output=output, delta=["12h"])

        _check_mars_requests(name, load_source.requests)

        missing_reference = False
        try:
            directory = get_test_archive(f"anemoi-datasets/create/mock-mars-{VERSION}/{name}.zarr.tgz")
        except requests.exceptions.HTTPError:
            missing_reference = True
            errors = [f"Reference data for {name} is missing, cannot compare."]

        if not missing_reference:
            from anemoi.datasets.commands.compare import compare_anemoi_datasets

            reference = os.path.join(directory, name + ".zarr")
            errors = compare_anemoi_datasets(reference=reference, actual=output, data=True)

        if errors or missing_reference:
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
