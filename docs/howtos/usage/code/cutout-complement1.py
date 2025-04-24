from anemoi.datasets import open_dataset

ds = open_dataset(
    complement={
        "cutout": ["lam-dataset", "global-dataset"],
        "min_distance_km": 1,
        "adjust": "dates",
    },
    source="global-dataset",
    interpolation="nearest",
)
