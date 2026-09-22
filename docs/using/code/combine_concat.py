from anemoi.datasets import open_dataset

# Two datasets covering consecutive periods, concatenated along the time
# axis into a single, longer dataset.
ds = open_dataset(
    concat=[
        "aifs-ea-an-oper-0001-mars-o96-1979-2000-6h-v8",
        "aifs-ea-an-oper-0001-mars-o96-2001-2022-6h-v8",
    ]
)
