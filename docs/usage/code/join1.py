from anemoi.datasets import open_dataset

ds = open_dataset("dataset1-1979-2022", "dataset2-1979-2022")

# or

ds = open_dataset(["dataset1-1979-2022", "dataset2-1979-2022"])

# or

ds = open_dataset(join=["dataset1-1979-2022", "dataset2-1979-2022"])
