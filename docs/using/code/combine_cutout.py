from anemoi.datasets import open_dataset

# A high-resolution regional dataset cut into a global one: the regional
# grid points replace the global ones wherever the two overlap.
ds = open_dataset(
    cutout=[
        "metno-regional-2p5km-2020-2022-6h-v1",
        "aifs-ea-an-oper-0001-mars-o96-2020-2022-6h-v8",
    ]
)
