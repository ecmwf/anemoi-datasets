from anemoi.datasets import open_dataset

ds = open_dataset(
    synthetic={
        "geography": {"bbox": [60, -10, 30, 20], "resolution": 0.25},
        "dates": {"start": "2020-01-01", "end": "2020-01-31", "frequency": "6h"},
        "layout": "gridded",
        "variables": [
            {"name": "2t", "values": {"constant": 273.15}},
            "msl",
            "insolation",
        ],
    }
)
