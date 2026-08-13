from anemoi.datasets import open_dataset

# Each call to open_dataset returns an object that behaves like a dataset.
# Because the result is itself a dataset, it can be passed to open_dataset
# again: the operations *compose*.

# 1. Open a dataset stored on disk (a leaf).
ds = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v8")

# 2. Subset it in time (wraps the leaf).
ds = open_dataset(ds, start=2000, end=2020)

# 3. Select a few variables (wraps the subset).
ds = open_dataset(ds, select=["2t", "msl", "10u", "10v"])

# 4. Change the frequency (wraps the selection).
ds = open_dataset(ds, frequency="12h")

# The result behaves exactly like the original dataset.
print(ds.shape)
print(ds.variables)
print(ds[0])
