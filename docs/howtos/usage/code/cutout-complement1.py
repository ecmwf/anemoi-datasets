from anemoi.datasets import open_dataset

ds = open_dataset(
    complement={
        "cutout": [
            "lam-dataset",
            {
                "dataset": "global-dataset",
                "select": ["tp"],
            },
        ],
        "min_distance_km": 1,
        "adjust": "dates",
    },
    source="global-dataset",
    interpolation="nearest",
)
