from anemoi.datasets import open_dataset

ds = open_dataset(
    {
        "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v8.zarr",
        "start": 2000,
        "end": 2020,
        "select": ["2t", "msl"],
        "frequency": "12h",
    }
)

# `tree()` returns the network of objects that was built. Printing it shows
# how the operations are nested, from the outermost operation down to the
# leaf (the zarr store).
print(ds.tree())
