from anemoi.datasets import add_dataset_path
from anemoi.datasets import open_dataset

add_dataset_path("https://object-store.os-api.cci1.ecmwf.int/ml-examples/")

ds = open_dataset("an-oper-2023-2023-2p5-6h-v1")
