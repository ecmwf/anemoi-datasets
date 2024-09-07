ds = open_dataset(dataset)

for i in range(len(ds) - 1):
    ds = ds[i + 1] - ds[i]
