#!/usr/bin/env python3
import argparse
import logging
import os
import shutil

import tqdm

from anemoi.datasets import open_dataset

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="open a dataset and build a new one")
    parser.add_argument("input", help="input dataset")
    parser.add_argument("output", help="output dataset")
    parser.add_argument("--backend", help="backend to use", type=str, default="npz1")
    parser.add_argument("--overwrite", help="overwrite output directory if it exists", action="store_true")
    args = parser.parse_args()
    build(**vars(args))


def build(input, output, backend, overwrite=False):
    ds = open_dataset(input, backend=backend)
    print(f"Using dataset {ds} as input")
    print(f"{input} backend is '{ds.metadata['backend']}'")
    print(f"Dataset has {len(ds)} records, from {ds.start_date} to {ds.end_date}")
    print(f"Converting dataset to {output} using new backend '{backend}'")

    from anemoi.datasets.data.records.backends import writer_backend_factory

    if os.path.exists(output):
        if overwrite:
            LOG.warning(f"Output directory {output} already exists, removing it")
            shutil.rmtree(output)
        else:
            raise FileExistsError(f"Output directory {output} already exists, use --overwrite to remove it")
    writer = writer_backend_factory(backend, output)

    for i in tqdm.tqdm(range(len(ds))):
        writer.write(i, ds[i])

    writer.write_statistics(ds.statistics)

    metadata = ds.metadata.copy()
    metadata["backend"] = backend
    writer.write_metadata(metadata)


if __name__ == "__main__":
    main()
