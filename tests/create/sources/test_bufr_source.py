import pytest

from anemoi.datasets.create.sources import create_source
from anemoi.datasets.create.sources.bufr import BUFRSource


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
                "per_report": {
                    "latitude": "latitude",
                    "longitude": "longitude",
                }
            },
        }
    }
    source = create_source(context=None, config=config)
    assert isinstance(source, BUFRSource)


def test_bufr_source_bad_config():
    config = {
        "bufr": {
            "source": {
                "invalid_source": {
                    "foo": "bar",
                }
            },
            "extract": {
                "per_report": {
                    "latitude": "latitude",
                    "longitude": "longitude",
                }
            },
        }
    }
    with pytest.raises(ValueError, match="Invalid source name"):
        _ = create_source(context=None, config=config)
