from anemoi.datasets import open_dataset

ds = open_dataset(
    dataset={
        "join": [
            {
                "dataset": "dataset-3h",
                "frequency": "3h",
            },
            {
                "dataset": "dataset-24h",
                "interpolate_frequency": "3h",
            },
        ],
        "adjust": "dates",
    },
    start="2004-01-01",
    end="2023-01-01",
)
