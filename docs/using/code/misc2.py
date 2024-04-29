from anemoi.datasets import add_named_dataset
from anemoi.datasets import open_dataset

add_named_dataset(
    "example-dataset",
    "https://object-store.os-api.cci1.ecmwf.int/ml-examples/an-oper-2023-2023-2p5-6h-v1.zarr",
)

ds = open_dataset("example-dataset")
