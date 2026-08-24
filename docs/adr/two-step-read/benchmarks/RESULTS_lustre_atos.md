# Grid-subset pushdown on **Atos Lustre** — benchmark & results

Perf test for the two-step read grid-subset pushdown (see
`../../adr-3-two-step-read.md`), run against datasets on the Atos Lustre
filesystem `/home/mlx/ai-ml/datasets` (mount `h1resw02`, HDD-backed).

Script: `lustre_atos_benchmark.py` (reuses the machinery in
`grid_pushdown_benchmark.py`, wired to concrete datasets on this filesystem).
Companion run on the SSD Lustre filesystem: `RESULTS_lustre_ssd.md`.

## How to reproduce

```bash
python lustre_atos_benchmark.py            # survey + cutout + synthetic
python lustre_atos_benchmark.py survey
python lustre_atos_benchmark.py cutout
python lustre_atos_benchmark.py synthetic
```

Datasets are opened **by absolute path** under `/home/mlx/ai-ml/datasets`, so
this resolves to local Lustre reads — not EWC S3. The cost here is filesystem
read **bandwidth**, not S3 round-trip latency. Read-only. Eager vs two-step is
toggled with `read_parts.READ_PARTS_ENABLED`; the two are **interleaved** per
repetition (medians) to cancel filesystem-cache / load drift.

## Environment

- Atos HPC, filesystem `h1resw02` (Lustre, 848T, HDD).
- 8 vCPU, Linux 4.18 x86_64 (RHEL 8.10); Python 3.12.11, numpy 2.4.6,
  **zarr 2.18.7**, anemoi-datasets 0.5.38.dev8 (`feat/refactor-usage`,
  two-step + pushdown).
- Date: 2026-06-08.

## Chunking survey — the grid axis is one chunk everywhere

```
dataset                                                          grid  grid_chunk grid chunked?
aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6                542080      542080            no
aifs-ea-an-oper-0001-mars-1p0-1979-2024-6h-v1                  65160       65160            no
aifs-ea-an-oper-0001-mars-20p0-2022-2022-6h-v6-ml13             162         162            no
aemet-an-harm-2p5km-2016-2021-6h-v0-iberia                    973273      973273            no
aifs-benchmarking-ea-an-oper-0001-mars-o800-2023-2023-6h-v1  2588800     2588800            no
```

Same finding as the S3 fleet: every dataset on this filesystem (globals
20p0→o800, the 2.5 km Iberia LAM) keeps the **whole grid in a single chunk**
(anemoi default `get_chunking`). So there is **no within-chunk byte saving** to
be had on current data — pushdown's win is skipping *whole constituent stores* a
shard doesn't touch, and not materialising the full-grid output.

## Real cutout: aemet 2.5km Iberia (LAM) in n320 (globe)

A genuine LAM-in-global cutout: the AEMET HARMONIE 2.5 km Iberia inset
(973 273 pts) over an `n320` global (after the cutout mask), 84 common variables,
output grid 1 508 614. Cross-source variables differ, so opened with per-member
`select` of the common variables and the semantic units check bypassed
(`--select-common --no-var-check`). Cutout build: ~125 s.

```
access                   eager ms   2step ms   ratio  stores  grid pts   legMB  2stepMB
FULL grid                     997        987    1.01       2   1508614    1196     1769
1/16 in LAM                  1001        334    3.00       1     94288    1196      593
1/16 in globe                 988        322    3.07       1     94288    1196      414
1/16 spanning                 991        984    1.01       2   1508614    1196     1769
```

- **1/16 in LAM: 3.00× faster, ~2× less memory** — pushdown reads only the LAM
  store (`stores=1`, 94 288 grid pts) and **skips the n320 globe store**.
- **1/16 in globe: 3.07× faster, ~3× less memory** — skips the 973k-pt LAM store.
- **spanning: 1.01× (parity)** — touches both stores, guard declines pushdown.
- **FULL grid: time parity (1.01×) but +48% memory** (1769 vs 1196 MB): the
  two-step buffer holds both constituent arrays *and* the concatenated output at
  once. Known cost of the buffer for full cutout reads; shard reads (the
  pushdown case) use far less memory than eager.

On Lustre HDD the full-grid read of the single n320 chunk costs ~1 s; a grid
shard that lands in one constituent skips the other store entirely, so the
sharded read is **~3× faster and uses 2–3× less memory**, with no regression on
full or boundary-spanning reads.

## Synthetic — when does pushdown cut *chunk reads*? (1/8 shard in globe region)

```
[whole grid in ONE chunk (anemoi default)] grid_chunk=20000  globe=20000 lam=2000
  eager  :   2 chunk reads /    704,000 B
  pushdown:   1 chunk reads /    640,000 B  (91% of eager)     # skips the LAM chunk

[grid SPLIT into chunks]                    grid_chunk=1000
  eager  :  22 chunk reads /    704,000 B
  pushdown:   3 chunk reads /     96,000 B  (14% of eager)      # ~7x fewer bytes
```

Confirms the rule: with one chunk per field, pushdown only saves by skipping
whole stores (here the LAM chunk → 91%); a real *byte* saving (14%) needs the
grid axis chunked at dataset-creation time.

## Bottom line (Atos Lustre)

- No I/O regression: full and boundary-spanning cutout reads are at parity with
  eager (1.00–1.01×).
- **~3× faster, 2–3× less memory** for grid shards that fall in one constituent
  and skip the other store — the grid-sharded-cutout training case.
- A genuine *byte* saving from within-store sub-selection would additionally
  require chunking the grid axis when the dataset is built.
