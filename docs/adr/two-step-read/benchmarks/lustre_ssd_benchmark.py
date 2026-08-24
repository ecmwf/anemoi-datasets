#!/usr/bin/env python
# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Two-step read grid-pushdown benchmark on the **Lustre SSD** filesystem.

Datasets live under ``/ec/ai/project/ai-ml/datasets`` (Lustre, SSD-backed, mount
``h1aiws01``). Opened **by absolute path** (not by name / S3). Because the SSD
flash array serves the full-grid read fast, this is the *hardest* case for the
refactor to win on time — yet store-skipping shards still cut both wall time and
peak memory, and there is never a regression on full reads.

Runs the same three modes as ``grid_pushdown_benchmark.py`` (which this imports),
wired to concrete datasets on this filesystem:

  python lustre_ssd_benchmark.py            # all three modes
  python lustre_ssd_benchmark.py survey
  python lustre_ssd_benchmark.py cutout
  python lustre_ssd_benchmark.py synthetic

See ``RESULTS_lustre_ssd.md`` for captured numbers and discussion.
"""

import sys
from pathlib import Path

# Import the shared benchmark machinery (survey/cutout/synthetic + helpers).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_pushdown_benchmark as bench  # noqa: E402

ROOT = "/ec/ai/project/ai-ml/datasets"

# Same cutout geometry as the Atos run for a like-for-like comparison: a 2.5 km
# Iberia LAM inside an n320 global, both single-chunk on the grid axis.
GLOBE = f"{ROOT}/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr"
LAM = f"{ROOT}/aemet-an-harm-2p5km-2016-2021-6h-v1-iberia.zarr"

# Representative spread of grids actually present on this filesystem.
SURVEY = [
    f"{ROOT}/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr",
    f"{ROOT}/aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8.zarr",
    f"{ROOT}/aifs-mc-an-oper-0001-mars-n128-2018-2026-3h-v2.zarr",
    f"{ROOT}/aemet-an-harm-2p5km-2016-2021-6h-v1-iberia.zarr",
    f"{ROOT}/aemet-an-harm-2p5km-2016-2021-6h-v0-canarias.zarr",
]


def main(mode: str) -> None:
    print("# Lustre SSD (h1aiws01) — two-step grid-pushdown benchmark\n")
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
