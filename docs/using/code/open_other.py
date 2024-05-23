from anemoi.datasets import open_dataset

ds1 = open_dataset("/path/to/dataset.zarr")

ds2 = open_dataset(ds1, frequency="24h", start="2000", end="2010")
