import yaml

from anemoi.datasets import open_dataset

with open("config.yaml") as file:
    config = yaml.safe_load(file)

ds = open_dataset(config)
