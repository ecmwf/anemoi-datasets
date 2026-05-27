# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from anemoi.utils.testing import GetTestData
from anemoi.utils.testing import skip_if_offline

from .utils.create import create_dataset
from .utils.mock_sources import LoadSource

HERE = Path(__file__).parent


@pytest.fixture
def load_source(get_test_data: GetTestData) -> LoadSource:
    return LoadSource(get_test_data)


@skip_if_offline
def test_missing_range_dates_are_supported(tmp_path: Path, load_source: LoadSource) -> None:
    recipe = HERE / "missing-range.yaml"
    output = tmp_path / "missing-range.zarr"

    with (
        patch("earthkit.data.from_source", load_source),
        patch("anemoi.datasets.create.sources.mars.from_source", load_source),
    ):
        create_dataset(recipe=str(recipe), output=str(output), delta=["12h"])

    dates = zarr.open(str(output), mode="r")["dates"][:]

    assert dates.tolist() == [
        np.datetime64("2020-12-30T00:00:00"),
        np.datetime64("2020-12-30T12:00:00"),
        np.datetime64("2021-01-03T00:00:00"),
        np.datetime64("2021-01-03T12:00:00"),
    ]


@skip_if_offline
def test_missing_mixture_dates_are_supported(tmp_path: Path, load_source: LoadSource) -> None:
    recipe = HERE / "missing-mixture.yaml"
    output = tmp_path / "missing-mixture.zarr"

    with (
        patch("earthkit.data.from_source", load_source),
        patch("anemoi.datasets.create.sources.mars.from_source", load_source),
    ):
        create_dataset(recipe=str(recipe), output=str(output), delta=["12h"])

    dates = zarr.open(str(output), mode="r")["dates"][:]

    assert dates.tolist() == [
        np.datetime64("2020-12-30T00:00:00"),
        np.datetime64("2020-12-31T12:00:00"),
        np.datetime64("2021-01-01T12:00:00"),
        np.datetime64("2021-01-03T12:00:00"),
    ]
