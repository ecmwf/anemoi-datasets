#!/usr/bin/env python
# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Two-step read grid-pushdown benchmark on the **Atos Lustre** filesystem.

Datasets live under ``/home/mlx/ai-ml/datasets`` (Lustre, HDD-backed, mount
``h1resw02``). Opened **by absolute path** (not by name / S3), so this measures
the pushdown win on a local-ish parallel filesystem where read bandwidth — not
S3 round-trip latency — is the cost.

Runs the same three modes as ``grid_pushdown_benchmark.py`` (which this imports),
wired to concrete datasets on this filesystem:

  python lustre_atos_benchmark.py            # all three modes
  python lustre_atos_benchmark.py survey
  python lustre_atos_benchmark.py cutout
  python lustre_atos_benchmark.py synthetic

See ``RESULTS_lustre_atos.md`` for captured numbers and discussion.
"""

import sys
from pathlib import Path

# Import the shared benchmark machinery (survey/cutout/synthetic + helpers).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pushdown_benchmark as bench  # noqa: E402

ROOT = "/home/mlx/ai-ml/datasets"

# Cutout pair on this filesystem: a 2.5 km regional LAM (Iberia) inside an n320
# global. Both single-chunk on the grid axis (anemoi default get_chunking), so
# the win is store-skipping + smaller output, not within-chunk byte savings.
GLOBE = f"{ROOT}/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr"
LAM = f"{ROOT}/aemet-an-harm-2p5km-2016-2021-6h-v0-iberia.zarr"

# Representative spread of grids actually present on this filesystem.
SURVEY = [
    f"{ROOT}/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr",
    f"{ROOT}/aifs-ea-an-oper-0001-mars-1p0-1979-2024-6h-v1.zarr",
    f"{ROOT}/aifs-ea-an-oper-0001-mars-20p0-2022-2022-6h-v6-ml13.zarr",
    f"{ROOT}/aemet-an-harm-2p5km-2016-2021-6h-v0-iberia.zarr",
    f"{ROOT}/aifs-benchmarking-ea-an-oper-0001-mars-o800-2023-2023-6h-v1.zarr",
]


def main(mode: str) -> None:
    print("# Atos Lustre (h1resw02) — two-step grid-pushdown benchmark\n")
    if mode in ("all", "survey"):
        print("## Chunking survey\n")
        bench.survey(SURVEY)
    if mode in ("all", "cutout"):
        print("\n## Real cutout: aemet 2.5km Iberia (LAM) in n320 (globe)\n")
        bench.cutout(LAM, GLOBE, n_shards=16, select_common=True, no_var_check=True)
    if mode in ("all", "synthetic"):
        print("\n## Synthetic chunk-read counting\n")
        bench.synthetic()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
