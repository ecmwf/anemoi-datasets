from anemoi.datasets import open_dataset

config = {
    "dataset": {
        "ensemble": [
            "/path/to/dataset1.zarr",
            {"dataset": "dataset_name", "end": 2010},
            {"dataset": "s3://path/to/dataset3.zarr", "start": 2000, "end": 2010},
        ],
        "frequency": "24h",
    },
    "select": ["2t", "msl"],
}

ds = open_dataset(config)
