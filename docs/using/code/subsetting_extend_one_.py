# Extend only backwards
ds = open_dataset(dataset, extend_start="2019-01-01")

# Extend only forwards
ds = open_dataset(dataset, extend_end="2021-12-31")
