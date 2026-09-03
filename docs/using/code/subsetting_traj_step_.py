# Select a single forecast step; returns a 4-D view
# (base_dates, variables, ensembles, cells) — shape-compatible
# with a gridded dataset at that lead time.
ds_t6 = open_dataset("traj.zarr", step=6)

# Select a list of steps; keeps the 5-D shape, narrows the step axis.
ds_subset = open_dataset("traj.zarr", steps=[6, 12, 18])

# Step range form (all three are optional).
ds_range = open_dataset("traj.zarr", step_start=6, step_end=24, step_frequency="6h")

# A step is a time-delta specification: a bare number means hours, and a
# suffixed string is honoured — so a sub-hourly dataset is addressed as:
ds_30m = open_dataset("traj.zarr", step="30m")
ds_sub = open_dataset("traj.zarr", step_start="10m", step_end="1h", step_frequency="10m")
