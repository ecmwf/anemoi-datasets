ds = open_dataset(
    dataset,
    skip_missing_dates=True,
    expected_access=slice(0, 2),
)

for i in range(len(ds)):
    xi, xi_1 = ds[i]
    dx = xi_1 - xi
