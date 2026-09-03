from anemoi.datasets import open_dataset

# The same network again, but with the operations passed as keyword
# arguments to a single call rather than collected in a dictionary.
ds = open_dataset(
    "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v8",
    start=2000,
    end=2020,
    select=["2t", "msl", "10u", "10v"],
    frequency="12h",
)

print(ds.shape)
