from anemoi.datasets import open_dataset

ds = open_dataset(
    {
        "dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v8.zarr",
        "start": 2000,
        "end": 2020,
        "select": ["2t", "msl"],
        "frequency": "12h",
    }
)

# The metadata records every operation applied to the data, the sources
# that were combined, the provenance of the run, and the supporting arrays
# (latitudes, longitudes, etc.). Anything that reads the dataset can later
# reconstruct exactly how it was produced.
metadata = ds.metadata()

print(sorted(metadata.keys()))
print(metadata["start_date"], metadata["end_date"])
print(metadata["variables"])
