from anemoi.datasets import open_dataset

ds = open_dataset("dataset-1979-2000", "dataset-2001-2022")

# or

ds = open_dataset(["dataset-1979-2000", "dataset-2001-2022"])

# or

ds = open_dataset(concat=["dataset-1979-2000", "dataset-2001-2022"])
