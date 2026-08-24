import yaml

from anemoi.datasets import open_dataset

# The whole network of operations can be stored as data (here YAML) and
# passed to open_dataset as a single dictionary. This is what allows a
# training configuration file to fully describe the training data.
with open("compose_config.yaml") as file:
    config = yaml.safe_load(file)

ds = open_dataset(config["dataset"])

print(ds.shape)
