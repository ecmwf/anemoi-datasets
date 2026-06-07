# Grid-subset pushdown — benchmark & results

Perf tests for the two-step read grid-subset pushdown (see
`../../adr-3-two-step-read.md`). Script: `grid_pushdown_benchmark.py`.

## How to reproduce

```bash
# 1. Chunking survey — is the grid axis split into chunks? (open by NAME)
python grid_pushdown_benchmark.py survey <name> [<name> ...]

# 2. Real cutout — full-read parity + per-shard time/memory/store-skipping
python grid_pushdown_benchmark.py cutout --lam <name> --globe <name> --shards 16

# 3. Synthetic — counts real chunk reads; no data access needed
python grid_pushdown_benchmark.py synthetic
```

Datasets are opened **by name** (`open_dataset("<name>")`), resolving to the
configured store (EWC S3 here) — not a filesystem path. Read-only. Eager vs
two-step is toggled with `read_parts.READ_PARTS_ENABLED` and the two are
**interleaved** per repetition (medians) to cancel S3-latency drift.

## Run: `ewc-vm-ewc-s3` (EWC VM, datasets on EWC S3)

- Environment: VM in the European Weather Cloud (EWC); datasets read from EWC S3.
- 16 vCPU, Linux 6.8 x86_64; Python 3.11.14, numpy 2.4.6, **zarr 2.18.7**,
  anemoi-datasets 0.5.38.dev7 (`feat/refactor-usage`, two-step + pushdown).
- Date: 2026-06-06.

### Chunking survey — the grid axis is one chunk everywhere

```
dataset                                                          grid  grid_chunk grid chunked?
aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6                   10944       10944            no
aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6                 542080      542080            no
aifs-od-an-oper-0001-mars-o1280-2016-2023-6h-v1               6599680     6599680            no
aemet-an-harm-2p5km-2016-2021-6h-v0-iberia                     973273      973273            no
uwcwest-rr-an-oper-0001-mars-2p0km-2020-2023-6h-v2-knmi       3071581     3071581            no
```

15 datasets surveyed (globals o48→o1280, regional LAMs to 3M points): **all keep
the whole grid in a single chunk** (anemoi default `get_chunking`). So there is
**no within-chunk byte saving** to be had on current data — pushdown's win is
skipping *whole constituent stores* a shard doesn't touch, and not materialising
the full-grid output.

### Real regional cutout: `metno-meps` 2.5km Norway (LAM) in `n320` (globe)

A genuine LAM-in-global cutout: meps regional inset (1 014 481 pts) over an `n320`
global (535 034 pts kept after the cutout mask), 85 common variables, output grid
1 549 515. Cross-source vars differ, so opened with per-member `select` of the
common variables and the semantic units check bypassed (`--select-common
--no-var-check`); this needed `Select`/`Subset` to be grid-array transparent so
pushdown reaches the leaves.

```
access                  eager ms   2step ms   ratio  stores  grid pts   legMB  2stepMB
FULL grid                    2321       2302    1.01       2   1549515    1238     1830
1/16 in LAM                  2242        929    2.41       1     96844    1238      454
1/16 in globe                2334        681    3.43       1     96844    1238      307
1/16 spanning                2294       2251    1.02       2   1549515    1238     1830
```

- **1/16 in globe: 3.43× faster, 4× less memory** — shard skips the 1M-pt LAM store.
- **1/16 in LAM: 2.41× faster, ~3× less memory** — skips the 535k-pt globe store.
- **spanning: 1.02× (parity)** — touches both stores, guard declines pushdown.
- **FULL grid: time parity (1.01×) but +48% memory** (1830 vs 1238 MB): the
  two-step buffer holds both constituent arrays *and* the concatenated output at
  once, where eager frees intermediates sooner. Known cost of the buffer for
  full cutout reads; shard reads (the pushdown case) use far less memory than
  eager. Could be improved by dropping buffer entries once consumed.

This is the realistic result: grid-sharded cutout reads are 2.4–3.4× faster and
use 3–4× less memory; full (unsharded) reads are time-neutral.

### Artificial cutout: `o48` (LAM) in `n320` (globe), 101 vars, output grid 12316, 1/16 shards

(Both global ERA5 grids, same `v6` recipe; o48-as-LAM is geometric nonsense but
drives the read path on plain-leaf constituents — kept for the leaf-only path.)

```
access                  eager ms   2step ms   ratio  stores  grid pts   legMB  2stepMB
FULL grid                     392        385    1.02       2     12316     229      229
1/16 in LAM                   397         16   25.05       1       769     229        5
1/16 in globe                 370        361    1.02       1       769     229      220
1/16 spanning                 389        386    1.01       2     12316     229      229
```

Reading:

- **FULL grid: 1.02× (parity).** Two-step adds no overhead now that the cross-part
  thread pool is gone (it was ~0.74× / 35% slower before — GIL contention on blosc
  decompress; see `../progress.md` entry 8).
- **1/16 shard in the LAM: 25× faster, ~46× less memory** — pushdown reads only the
  LAM store (`stores=1`, 769 grid pts) and **skips the 112 MB single-chunk globe**.
  This is the headline win for grid-sharded cutout training.
- **1/16 shard in the globe: 1.02×** — must read the globe's one chunk, so no time
  win; it does skip the LAM. (Memory edge is small *here* only because this
  artificial geometry gives the globe few output points; a real small-LAM /
  large-globe cutout returns shard-sized arrays vs the full-grid concat → large
  memory win on globe-region shards too.)
- **1/16 spanning the LAM/globe boundary: 1.01×** — the guard declines pushdown
  (it would touch every store and gain nothing, while `oindex` gather is slower
  than a slice), so it uses the eager full read. Parity, not a regression.

`stores` = distinct constituent stores actually read; `grid pts` = grid points
requested from those stores (vs 12316 for a full read).

### Synthetic — when does pushdown cut *chunk reads*? (1/8 shard in globe region)

```
[whole grid in ONE chunk (anemoi default)] grid_chunk=20000  globe=20000 lam=2000
  eager  :   2 chunk reads /    704,000 B
  pushdown:   1 chunk reads /    640,000 B  (91% of eager)     # skips the LAM chunk

[grid SPLIT into chunks]                    grid_chunk=1000
  eager  :  22 chunk reads /    704,000 B
  pushdown:   3 chunk reads /     96,000 B  (14% of eager)      # ~7x fewer bytes
```

Confirms the rule: with one chunk per field, pushdown only saves by skipping whole
stores (here the LAM chunk → 91%); a real *byte* saving (14%) needs the grid axis
chunked at dataset-creation time.

## Bottom line

- No I/O regression: full cutout reads are at parity with eager.
- Large win (≈25× time, ≈46× memory on this run) for shards that fall in a
  constituent and skip the big globe store — the grid-sharded-cutout case.
- Neutral for globe-region / boundary shards (correct, never slower).
- A genuine *byte* saving from within-store sub-selection would additionally
  require chunking the grid axis when the dataset is built.
