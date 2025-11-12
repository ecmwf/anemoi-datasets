# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytest

from anemoi.datasets.create.sources import create_source

LOG = logging.getLogger(__name__)


def test_csv_source_registration():

    source = create_source(context=None, config={"csv": {"path": "data.csv"}})

    with pytest.raises(NotImplementedError):
        source.execute(dates=[])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_csv_source_registration()
