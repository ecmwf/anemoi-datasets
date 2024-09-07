ds = open_dataset(
    dataset,
    skip_missing_dates=True,
    expected_access=slice(0, 2),
)

for i in range(len(ds) - 1):
    ds = ds[i + 1] - ds[i]
