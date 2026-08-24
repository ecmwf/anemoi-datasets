values = ds[0]
normalised = (values - dataset.statistics["mean"]) / dataset.statistics["stdev"]
