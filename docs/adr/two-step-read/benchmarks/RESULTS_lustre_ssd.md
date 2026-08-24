# Grid-subset pushdown on **Lustre SSD** — benchmark & results

Perf test for the two-step read grid-subset pushdown (see
`../../adr-3-two-step-read.md`), run against datasets on the SSD-backed Lustre
filesystem `/ec/ai/project/ai-ml/datasets` (mount `h1aiws01`).

Script: `lustre_ssd_benchmark.py` (reuses the machinery in
`grid_pushdown_benchmark.py`, wired to concrete datasets on this filesystem).
Companion run on the HDD Lustre filesystem: `RESULTS_lustre_atos.md`. Same
cutout geometry on both, for a like-for-like comparison.

## How to reproduce

```bash
python lustre_ssd_benchmark.py            # survey + cutout + synthetic
python lustre_ssd_benchmark.py survey
python lustre_ssd_benchmark.py cutout
python lustre_ssd_benchmark.py synthetic
```

Datasets are opened **by absolute path** under `/ec/ai/project/ai-ml/datasets`,
so this resolves to local Lustre reads — not EWC S3. This is the SSD flash
array, the *hardest* case for the refactor to win on time: the full-grid read is
already fast. Read-only. Eager vs two-step is toggled with
`read_parts.READ_PARTS_ENABLED`; the two are **interleaved** per repetition
(medians) to cancel filesystem-cache / load drift.

## Environment

- Atos AI partition, filesystem `h1aiws01` (Lustre, 3.3P, SSD).
- 8 vCPU, Linux 4.18 x86_64 (RHEL 8.10); Python 3.12.11, numpy 2.4.6,
  **zarr 2.18.7**, anemoi-datasets 0.5.38.dev8 (`feat/refactor-usage`,
  two-step + pushdown).
- Date: 2026-06-08.

## Chunking survey — the grid axis is one chunk everywhere

```
dataset                                                          grid  grid_chunk grid chunked?
aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6                542080      542080            no
aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8                542080      542080            no
aifs-mc-an-oper-0001-mars-n128-2018-2026-3h-v2                 88838       88838            no
aemet-an-harm-2p5km-2016-2021-6h-v1-iberia                    973273      973273            no
aemet-an-harm-2p5km-2016-2021-6h-v0-canarias                  264985      264985            no
```

Same finding as the S3 fleet and the Atos HDD run: every dataset keeps the
**whole grid in a single chunk** (anemoi default `get_chunking`). So there is
**no within-chunk byte saving** to be had on current data — pushdown's win is
skipping *whole constituent stores* a shard doesn't touch, and not materialising
the full-grid output.

## Real cutout: aemet 2.5km Iberia (LAM) in n320 (globe)

Same geometry as the Atos run: AEMET HARMONIE 2.5 km Iberia inset (973 273 pts)
over an `n320` global, 84 common variables, output grid 1 508 614. Opened with
per-member `select` of the common variables and the semantic units check
bypassed (`--select-common --no-var-check`). Cutout build: ~112 s.

```
access                   eager ms   2step ms   ratio  stores  grid pts   legMB  2stepMB
FULL grid                    1016       1012    1.00       2   1508614    1196     1773
1/16 in LAM                  1023        333    3.07       1     94288    1196      599
1/16 in globe                 918        318    2.89       1     94288    1196      414
1/16 spanning                 918        921    1.00       2   1508614    1196     1773
```

- **1/16 in LAM: 3.07× faster, ~2× less memory** — pushdown reads only the LAM
  store (`stores=1`, 94 288 grid pts) and **skips the n320 globe store**.
- **1/16 in globe: 2.89× faster, ~3× less memory** — skips the 973k-pt LAM store.
- **spanning: 1.00× (parity)** — touches both stores, guard declines pushdown.
- **FULL grid: time parity (1.00×) but +48% memory** (1773 vs 1196 MB): the
  two-step buffer holds both constituent arrays *and* the concatenated output at
  once. Known cost of the buffer for full cutout reads; shard reads (the
  pushdown case) use far less memory than eager.

The headline: even on a **fast SSD array**, where the full read is already ~1 s
and there is no S3 latency to hide behind, a grid shard that lands in one
constituent is **~3× faster and uses 2–3× less memory**, because it skips the
other constituent store entirely. Full and boundary-spanning reads are at
parity — no regression.

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

## Bottom line (Lustre SSD)

- No I/O regression: full and boundary-spanning cutout reads are at parity with
  eager (1.00×) even on fast flash.
- **~3× faster, 2–3× less memory** for grid shards that fall in one constituent
  and skip the other store — the grid-sharded-cutout training case. The win is
  filesystem-independent: it comes from *not issuing* the skipped store's reads,
  not from any storage-latency effect (compare `RESULTS_lustre_atos.md`: HDD
  3.00–3.07×, SSD 2.89–3.07×).
- A genuine *byte* saving from within-store sub-selection would additionally
  require chunking the grid axis when the dataset is built.
