#!/usr/bin/env python3
# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
