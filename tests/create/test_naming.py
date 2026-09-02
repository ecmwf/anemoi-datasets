# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.datasets.create.naming import check_dataset_name


@pytest.mark.parametrize(
    "name,valid",
    [
        ("aifs-ea-an-oper-0001-mars-o96-2010-2010-6h-v1", True),
        ("aifs-od-an-oper-0001-mars-n320-2010-2010-6h-v1", True),
        ("aifs-od-an-oper-0001-mars-0p25-2010-2010-6h-v1", True),
        ("aifs-od-an-oper-0001-mars-2p5km-2010-2010-6h-v1", True),
        ("dwd-dream-archive-r03b03-2010-2024-3h-v1-ml14", True),
        ("aifs-od-an-oper-0001-mars-r3b3-2010-2010-6h-v1", False),
        ("aifs-od-an-oper-0001-mars-o96-2010-2010-6x-v1", False),
        ("aifs_od_an_oper_0001_mars_o96_2010_2010_6h_v1", False),
        ("AIFS-od-an-oper-0001-mars-o96-2010-2010-6h-v1", False),
        ("aifs-od-an-oper-0001-mars-o96-2010-2010-6h", False),
    ],
)
def test_check_dataset_name_examples(name: str, valid: bool) -> None:
    messages = check_dataset_name(name)
    if valid:
        assert messages == []
    else:
        assert messages != []
