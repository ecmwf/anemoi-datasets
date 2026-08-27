ds = open_dataset(dataset, residual_statistics="residual.json")

# Will return the statistics of the difference between the two
# datasets recorded in "residual.json"

print(ds.residual_statistics)

# The provenance of the file is kept in the dataset metadata

print(ds.metadata()["specific"]["residual_statistics"]["datasets"])
