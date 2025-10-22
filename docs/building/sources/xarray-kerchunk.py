import json

import fsspec
import tqdm
from kerchunk.combine import MultiZarrToZarr
from kerchunk.hdf import SingleHdf5ToZarr

fs = fsspec.filesystem("s3", anon=True)

pattern = "s3://nsf-ncar-era5/e5.oper.an.pl/202403/e5.oper.an.pl.*.ll025sc.2024????00_2024????23.nc"


jsons = []

for file in tqdm.tqdm(fs.glob(pattern)):
    with fs.open(file, "rb", anon=True) as f:
        h5chunks = SingleHdf5ToZarr(f, file)
        jsons.append(h5chunks.translate())


mzz = MultiZarrToZarr(
    jsons,
    remote_protocol="s3",
    remote_options={"anon": True},
    concat_dims=["time"],
    identical_dims=["latitude", "longitude"],
)

with open("combined.json", "w") as f:
    json.dump(mzz.translate(), f)
