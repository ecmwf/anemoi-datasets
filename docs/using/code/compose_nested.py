from anemoi.datasets import open_dataset

# The same network of operations as in `compose_steps.py`, but described
# in a single, nested call. The operations are applied from the inside
# out: open -> subset -> select -> change frequency.
ds = open_dataset(
    {
        "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v8",
        "start": 2000,
        "end": 2020,
        "select": ["2t", "msl", "10u", "10v"],
        "frequency": "12h",
    }
)

print(ds.shape)
