#!/usr/bin/env python3
# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging
import os

from anemoi.utils.remote import transfer

LOG = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Upload sample dataset to S3")
parser.add_argument("--bucket", type=str, help="S3 target path", default="s3://ml-tests/test-data/")
parser.add_argument("source", type=str, help="Path to the sample dataset")
parser.add_argument("target", type=str, help="Path to the sample dataset")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data")

args = parser.parse_args()

source = args.source
target = args.target
bucket = args.bucket

assert os.path.exists(source), f"Source {source} does not exist"

if not target.startswith("s3://"):
    if target.startswith("/"):
        target = target[1:]
    if bucket.endswith("/"):
        bucket = bucket[:-1]
    target = os.path.join(bucket, target)

LOG.info(f"Uploading {source} to {target}")
transfer(source, target, overwrite=args.overwrite)
LOG.info("Upload complete")
