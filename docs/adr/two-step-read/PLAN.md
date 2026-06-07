# Plan — two-step read & multi-cutout reads

How we reach the objectives in `GOAL.md`. Status of each step lives in
`progress.md` (this file is the route, that file is the odometer).

## Workstream A — Documentation & decision ✅ DONE

- [x] A1. Audit the shipped code vs `adr-3-two-step-read.md`; list divergences.
- [x] A2. Rewrite ADR status + add "Implementation as built", "Why the eager
  path is permanent", revised Phase 4/5, "Next steps".
- [x] A3. ✅ Done — `test_eager_only_wrapper_produces_transformed_data_via_fallback`
  (verify): a `rolling_average` produces its *transformed* result via fallback
  (two-step == eager, and ≠ the raw centre row) → deleting its `__getitem__`
  fails the test. Plus an **oracle** (`test_two_step_oracle.py`) asserting two-step
  never silently diverges from eager across all dataset types × edge indices
  (OOB / negative / empty / step / negstep / spanning / over-stop / per-axis int).
- [x] A4. ✅ Done — `ANEMOI_DATASETS_READ_PARTS` is now a real kill-switch:
  default on, set `=0`/`false`/`no` to force the eager path. The gate reads the
  module attr each call (runtime-toggleable; tests use it for eager oracle).

## Workstream B — Correctness oracle hardening ✅ DONE

- [x] B1. Wrapper support map in `progress.md` (two-step vs eager-only); each
  two-step wrapper has a matching `read_from_buffer`.
- [x] B2. Byte-identity oracle: `test_two_step_oracle.py` (all dataset types ×
  edge indices) + `phase2/3/4/5` + `verify` assert `two_step == eager`.
- [x] B3. Per-node fallback covered: `gather_parts` returns `None` if any child is
  unsupported (e.g. `concat([ok, rolling_average])`), gate falls back; the
  supported subtrees still two-step (`test_read_parts_phase4` + oracle).

## Workstream C — `multi=` container ✅ DONE (2026-06-06)

- [x] C1. API decided: `open_dataset(multi=dict(name=spec, ...))` → a `Multi`
  dataset; `ds[n]` returns `{name: array}`. Written up in the ADR ("Problem 2").
- [x] C2. Added the `multi` branch to `usage/misc.py::_open_dataset` and
  `multi_factory` + `Multi` in `usage/gridded/multi.py`.
- [x] C3. Shared stores resolve to a **single** `ZarrStore`/`zarr.Array` via
  `shared_zarr_opens()` + open-cache (contextvar) in `usage/store.py`, consulted
  by `ZarrStore.from_name_or_path`. `factorize` (keyed on `id(self.data)`) then
  dedups the shared global across cutouts.
- [x] C4. Confirmed in `tests/test_multi.py`: 2 cutouts over a shared global
  factorize to **3 reads, not 4** (N+1 vs 2N). Contrast test shows independent
  opens give 4. (Wall-clock S3 benchmark still nice-to-have, not blocking.)

## Workstream D — Grid-subset pushdown (the big I/O win) ✅ DONE (D1–D7)

- [x] D1. ✅ `ReadPart.grid_index` (optional int tuple on the last axis); identity
  includes it. Rectangular fast path unchanged.
- [x] D2. ✅ Union factorization — `factorize` groups grid-index parts by
  `(id(data), slices[1:])` and reads the **sorted union** of their grid points
  once; `mapping` carries `grid_cols` and `ReadBuffer` applies them. A shared
  global across cutouts (in a `multi=`) is read once even with different
  per-cutout grid subsets. Test: `TestMultiCutoutGridUnion`.
- [x] D3. ✅ `ReadPart.execute` uses `data.oindex[..., grid_index]` (zarr
  orthogonal indexing) — only the needed grid points are read.
- [x] D4. ✅ `Cutout._pushdown_plan` / `_grid_pushdown` map the output range
  through the (full-LAM + masked-globe) layout to per-store grid indices and emit
  grid-index parts; `split_grid_index` lets the leaf accept a grid-axis array.
  Guarded to non-full slices, step ≥ 1, consistent single-/non-overlapping-LAM.
- [x] D5. ✅ `tests/test_grid_pushdown.py`: byte-identity vs eager (within LAM,
  across boundary, within globe, single point, strided) + reads exactly the
  requested number of grid points.
- [x] D6. ✅ Wrapper transparency + safety. `Select` (var) and `Subset` (date) pass
  a grid-axis index array through (`split_grid_index` in `indexing.py`) so pushdown
  reaches the leaf in **real** cutouts (members opened with `select`/`adjust`).
  `Cutout._pushdown_supported` probes constituents (I/O-free) and falls back to the
  eager full read if any can't take a grid array — fixes a crash where the array
  raised `ValueError` past the gate. Validated on real `metno-meps`→`n320` cutout
  (2.4–3.4× faster shards); see `benchmarks/RESULTS.md`. Test:
  `TestPushdownThroughWrappers`.
- [x] D7. ✅ Store-skip guard conditioned on grid chunking. `Cutout._grid_is_chunked`
  (via forwarded `Forwards.chunks`): one-chunk grid → keep store-skip guard;
  chunked grid → push down even when all stores touched (oindex reads only touched
  chunks). Test: `TestChunkedGridGuard`. (Untestable on real data until a
  grid-chunked dataset exists.)

## Sequencing / dependencies

```
A1,A2 done ─► A3,A4 (cheap, do next)
            └► B1 ─► B2 ─► B3   (safety net before touching read code)
                         └► C1 ─► C2 ─► C3 ─► C4
                                     D1 ─► D2 ─► D3 ─► D4 ─► D5
```
- Do **B (oracle)** before **C/D** — never refactor the reader without the
  byte-identity net in place.
- C3 (single shared array object) and D1 (index-array ReadPart) are the two
  load-bearing technical risks; spike them early.

## Open questions — RESOLVED

- `multi=` members are **eagerly opened** in `multi_factory` (inside
  `shared_zarr_opens`); per-member options go inside each member spec; per-member
  indexing via `ds[{name: index}]`.
- Orthogonal indexing reads whole chunks → on the **one-chunk-per-field** real
  datasets there is no within-chunk byte saving; the win is skipping whole stores
  (+ memory). A real byte saving needs the grid axis chunked at creation. The
  store-skip guard is conditioned on `_grid_is_chunked` (D7). See `benchmarks/`.
- Fancy indices (list/array on a non-grid axis) **fall back** (leaf/`two_step_read`
  return `None`); no double-handling with `expand_list_indexing` (eager path).

## Remaining (not blocking)

- +48% peak memory on full cutout reads (documented limitation in the ADR).
- Optional: re-introduce cross-part concurrency as an explicit, measured option for
  latency-bound many-small-part workloads (removed as a global default).
