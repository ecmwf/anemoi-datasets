#!/usr/bin/env python3


import argparse
import os
import shutil

import xarray as xr

parser = argparse.ArgumentParser(description="Create a sample dataset")
parser.add_argument("input", type=str, help="Input file name")
parser.add_argument("output", type=str, help="Output file name")
args = parser.parse_args()

if os.path.exists(args.output):
    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    else:
        os.unlink(args.output)

if args.input.endswith(".zarr"):
    ds = xr.open_zarr(args.input)
else:
    ds = xr.open_dataset(args.input)

if args.output.endswith(".zarr"):
    ds.to_zarr(args.output, consolidated=True)
else:
    ds.to_netcdf(args.output)
