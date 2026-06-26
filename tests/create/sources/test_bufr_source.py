# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest
from pydantic import ValidationError

from anemoi.datasets.create.sources import create_source
from anemoi.datasets.create.sources.bufr import BUFRSource


def _base_recipe(bufr_config: dict) -> dict:
    return {
        "dates": {"values": [datetime.datetime(2026, 1, 1, 0, 0, 0)]},
        "input": bufr_config,
    }


def test_bufr_schema_accepts_mars_embedded_source():
    recipe = _base_recipe(
        {
            "bufr": {
                "source": {
                    "mars": {
                        "class": "od",
                    }
                },
                "extract": {
                    "latitude": "latitude",
                    "longitude": "longitude",
                    "per_report": {
                        "azimuth": "bearingOrAzimuth",
                    },
                },
            }
        }
    )
    from anemoi.datasets.create.recipe import Recipe

    parsed = Recipe(**recipe)
    assert parsed.input is not None


def test_bufr_schema_rejects_non_mars_embedded_source():
    recipe = _base_recipe(
        {
            "bufr": {
                "source": {"invalid_source": {"path": "/some/path"}},
                "extract": {
                    "latitude": "latitude",
                    "longitude": "longitude",
                    "per_report": {
                        "azimuth": "bearingOrAzimuth",
                    },
                },
            }
        }
    )

    with pytest.raises(ValidationError):
        from anemoi.datasets.create.recipe import Recipe

        Recipe(**recipe)


def test_bufr_source_mars_config():
    config = {
        "bufr": {
            "source": {
                "mars": {
                    "class": "od",
                    "expver": "0001",
                    "stream": "DCDA/LWDA",
                    "type": "ai",
                    "obstype": "atov",
                    "time": "00",
                }
            },
            "extract": {
                "latitude": "latitude",
                "longitude": "longitude",
                "per_report": {
                    "azimuth": "bearingOrAzimuth",
                },
            },
        }
    }
    source = create_source(context=None, config=config)
    assert isinstance(source, BUFRSource)


def test_bufr_source_rejects_coordinate_names_in_per_report():
    config = {
        "bufr": {
            "source": {
                "mars": {
                    "class": "od",
                }
            },
            "extract": {
                "latitude": "lat",
                "longitude": "lon",
                "per_report": {
                    "latitude": "someOtherLatKey",
                },
            },
        }
    }
    with pytest.raises(ValueError, match="reserved coordinate column names"):
        _ = create_source(context=None, config=config)
