# Progress — two-step read & multi-cutout reads

Live log for the effort described in `GOAL.md` / `PLAN.md`. Newest entry on top.
Keep this updated whenever the code or the ADR moves.

## Status at a glance

| Workstream | State |
|---|---|
| A. Docs & decision | A1, A2 ✅ · A3 ✅ (guard + oracle) · A4 ✅ (kill-switch) |
| B. Oracle hardening | ✅ **covered** by phase2–5 + verify (byte-identity oracle exists) |
| C. `multi=` container | ✅ **done** (impl + tests + ADR) |
| D. Grid-subset pushdown | ✅ **done** (D1–D5, incl. D2 union-factorize) |

Branch: `feat/refactor-usage` (anemoi-datasets).

## Wrapper support map (verified 2026-06-06)

Where two-step lives in the code and which wrappers fall back to eager.

Two-step **implemented** (rectangular reads):
- `ZarrStore` / `GriddedZarr` (leaf) — `usage/store.py:230` builds the `ReadPart`.
  Falls back to eager for list/array/boolean (fancy) indices.
- `ZarrWithMissingDates` — validates missing then delegates (`gridded/store.py:309`).
- `Forwards` (default pass-through) and the simple transforms it covers.
- `Select`, `Subset`, `Concat` (`gridded/concat.py`), `Cutout` & `GridsBase`
  (`gridded/grids.py`), `Join` (`gridded/join.py`), `Merge` (`gridded/merge.py`),
  `Ensemble` (`gridded/ensemble.py`), `Masked` (`gridded/masked.py`).
- `MissingDates` placeholder returns `[]` parts (no read) — `gridded/missing.py:449`.

Eager-**only** (`collect_read_parts` returns `None` → gate falls back — *by
design, permanent*):
- `InterpolateFrequency`, `InterpolateNearest` (`gridded/interpolate.py`)
- `RollingAverage` (`gridded/rolling_average.py`)
- `MissingDatesFill` (`gridded/fill_missing.py`), `SkipMissingDates` (`gridded/missing.py:322`)
- `ZipBase` (`gridded/xy.py`), `Chain` (`gridded/unchecked.py`)
- `Complement` (`gridded/complement.py`)
- Tabular: `tabular/store.py`, `tabular/select.py`, `tabular/tensors.py`
- `trajectories/subset.py`

## Key facts (current — reflects code after all entries below)

- Wiring is **always-on with per-node fallback** via `Dataset.__init_subclass__`.
  Fallback is signalled by **`collect_read_parts` returning `None`** (a normal
  outcome), not by an exception; the gate does `_f(self,n) if result is None`.
- **Kill-switch**: `ANEMOI_DATASETS_READ_PARTS=0` (read each call via
  `read_parts.READ_PARTS_ENABLED`) forces eager. Default on. (A4.)
- `ReadPart` holds the **live `zarr.Array`** + a label `path`; identity is
  `(id(self.data), slices, grid_index)`. ⇒ independent `open_dataset` calls on the
  same URL do **not** factorize together — hence `shared_zarr_opens` for `multi=`.
- `ReadPart` carries an optional **`grid_index`** (int tuple on the last axis),
  executed via `data.oindex[...]` → grid-subset pushdown (D). `factorize` unions
  grid indices of parts sharing a store (D2); `ReadBuffer` mapping is a 3-tuple
  `(merged, row_offset, grid_cols)`.
- **Cutout** pushes a grid shard down to per-store grid indices (`_grid_pushdown`),
  skipping constituent stores a shard doesn't touch; guarded by `_pushdown_supported`
  (constituents can take a grid array; `Select`/`Subset` are grid-transparent) and
  `_grid_is_chunked` (store-skip guard only for one-chunk grids). Falls back to the
  full read otherwise.
- `execute_parts` is **sequential** (cross-part thread pool removed — GIL-bound on
  blosc decompress; within-array S3 fetch already parallel in `getitems`).
- Leaf validates integer bounds (`check_int_bounds`) before normalising →
  `IndexError` for out-of-range ints (no silent `i % size` wrap).

## Log

### 2026-06-07 (17) — rename "legacy" → "eager" (the path is permanent, not deprecated)

User: "legacy" wrongly implies deprecated — the recursive `__getitem__` path is
**permanent**. Renamed the concept **`legacy` → `eager`** (reads zarr immediately
at the leaf). Pair: **"two-step read" ↔ "eager read"**. Kept "two-step" (accurate,
the feature identity; "delayed/deferred" would be wrong — it plans-then-executes
within one call, not lazily).

Scope: ~205 occurrences across src (32), tests (60, incl. helpers `eager_path()`,
`_eager()`, test names `test_eager_*`), docs (113), benchmark script, and the
memory files (`two-step-read-eager-path-permanent.md` + MEMORY index + `[[links]]`).
Pure rename, no behaviour change. 0 `legacy` left. Read-layer **183 passed**.

### 2026-06-07 (16) — doc/ADR/plan/progress consistency sweep

Final cleanup: aligned all docs with the code (no code change). Found stale refs
**only in docs** (code clean):
- `_split_grid_index` → `split_grid_index` (moved to `indexing.py`) in ADR/PLAN/progress.
- progress "Wrapper support map": eager-only wrappers now *return `None`* (not raise).
- progress "Key facts": rewrote the initial-audit snapshot (was: vestigial env var,
  rectangular-only ReadPart, cutout-over-reads, threaded execute) → current reality
  (kill-switch, `grid_index`/oindex, pushdown + guards, sequential execute,
  `check_int_bounds`).
- ADR: Status → "implemented, 2026-06-07"; `ReadBuffer` sketch → 3-tuple mapping;
  ABC `collect_read_parts` sketch → `return None`; migration Phase-4 env var →
  kill-switch (not vestigial).
- PLAN: A/B/D marked DONE; open questions marked RESOLVED; Multi dict API noted.
- GOAL DoD(4) corrected: sharded-cutout win is store-skip + memory; within-chunk
  byte saving only with a grid-chunked dataset (honest, matches benchmark).

Cross-checked ADR claims against code (base collect→None, gate None→eager,
`check_int_bounds`, `two_step_read` None, `ReadBuffer` 3-tuple, `Multi._member_indices`)
— all match. Read-layer suite **183 passed**.

### 2026-06-07 (15) — `multi[{name: index}]` per-member API + memory limitation noted

- **`multi` per-member index dict** (replaces the surprising broadcast-grid-slice):
  `ds[{"a": idx_a, "b": idx_b}]` indexes each member by its own index — the explicit
  way to grid-shard a multi (members have different grids). Broadcast `ds[t]` kept
  for shared-date reads. I/O still shared (one factorize pass), so a store common to
  members is read once even with *different* per-member shards (verified). Keys must
  be members; subset allowed; unknown key → KeyError. `_member_indices` in
  `multi.py`; docstring updated. Tests: `TestMultiDictIndex` (6). Read-layer **172
  passed** (+ multi 15).
- **+48% memory on FULL cutout reads** recorded in the ADR as a known limitation
  (not addressed): two-step `ReadBuffer` holds constituent arrays + concat at peak;
  cutout-specific; memory not correctness; sharded reads use less. Follow-up options
  listed (drop consumed buffer entries / route full cutout to eager with multi
  carve-out).

### 2026-06-07 (14) — A3 done + corner-case divergence hunt (trust rebuild)

After the OOB silent-wrap miss, hunted systematically for *other* silent
divergences (two-step gives wrong data instead of matching eager or raising).

Method: compare two-step vs eager (kill-switch) across **all** dataset types
(plain, Select, Subset, Concat, Join, Merge, Grids/GivenAxis, Cutout) × an
edge-index matrix: OOB ints (bare + per-axis in tuple), negatives, empty slices,
strided, **negative step**, cutout grid shards (within-LAM / spanning / within-
globe / int / over-stop). **Result: zero divergence** — same value and same
raise-behaviour everywhere. The OOB fix propagates to all leaves (trajectories
leaf inherits the fixed `ZarrStore.collect` with `check_int_bounds`).

Locked it as a committed oracle: `tests/test_two_step_oracle.py` (8 tests). Any
future silent divergence now fails CI.

A3 done: `test_eager_only_wrapper_produces_transformed_data_via_fallback` pins
that a eager-only wrapper (`rolling_average`) still yields its transformed result
through fallback (≠ raw centre row) → its `__getitem__` can't be silently deleted.

A4 was already done (kill-switch). Both A3+A4 ✅. Read-layer **176 passed**.

### 2026-06-07 (13) — fallback by None (not exception) + OOB int fix

Acted on two REVIEW findings.

**Fallback is now a return value, not an exception.** `collect_read_parts` returns
``None`` to mean "use eager" (a normal outcome). Changes:
- base `Dataset.collect_read_parts` → `return None` (default unsupported); the 14
  unsupported wrappers (interpolate ×2, rolling, fill/skip-missing, zip, chain,
  complement, tabular ×3, trajectory subsets ×3) `return None` instead of `raise
  NotImplementedError`.
- new `read_parts.gather_parts(children)` — flattens child results, returns ``None``
  if any child is ``None``. Used by Concat, Join, Merge, GivenAxis, Grids, Cutout,
  Multi (so e.g. `concat([ok, rolling_average])` falls back cleanly).
- gate: `result = two_step_read(...); return _f if result is None else result`
  (was `except NotImplementedError`). `two_step_read` returns ``None`` on
  `parts is None`; keeps a narrow `(ValueError, TypeError, AttributeError)` catch
  only for non-normalisable fancy indices.
- Tests updated: verify/phase3/phase4 now assert ``None`` / fallback, not raises.

**OOB int fix (the "very bad" silent wrap).** `ds[(n_dates, …)]` returned date 0
under two-step (`index_to_slices` does `i % size`). Added `check_int_bounds`
(indexing.py), called in the leaf before normalising → `IndexError` (matches
eager), for ints bare or in a tuple. Test added.

Regression: read-layer **167 passed**, `test_data` **49 passed**. ADR wiring
section + REVIEW updated.

### 2026-06-07 (12) — fresh soundness review (see REVIEW.md)

Independent audit of the whole branch (core math, gate, leaf, Select/Subset,
Cutout pushdown, Multi, decisions, tests). **No correctness bug found in the new
code**; collect↔read mirror, factorize grid-union, ReadPart identity round-trip all
verified sound. Decisions defensible, claims honest. Findings in `REVIEW.md`:
- 🟠 only material concern: **+48% memory on FULL cutout reads** (default path) —
  buffer holds constituent arrays + concat at once. Cutout-specific.
- 🟡 fallback is `NotImplementedError`-typed (contract); tuple int-date OOB
  modulo-wraps vs eager `BoundsCheckError` (pre-existing, verified); `multi[...]`
  grid-index has per-member semantics.
Read-layer suite **164 passed**.

### 2026-06-07 (11) — store-skip guard conditioned on grid chunking + ReadCache→ReadBuffer rename

- Renamed `ReadCache`→`ReadBuffer`, `read_from_cache`→`read_from_buffer`, the
  `cache` var/param→`buffer` (per-read buffer, not a cache). Kept genuine caches:
  `shared_zarr_opens` open-cache, zarr `LRUStoreCache`, `cached_property`.
- **Store-skip guard now conditioned on grid chunking.** `Cutout._grid_is_chunked`
  (uses `chunks`, now forwarded through `Forwards`): grid one-chunk/field (today's
  reality) → keep store-skip guard (spanning/all-store shards fall back; oindex
  gather slower, no within-chunk saving); grid chunked → push down even when every
  store is touched (oindex reads only touched chunks → real within-store saving).
  Implements the chunked-dataset extrapolation. Tests: `TestChunkedGridGuard`.
- Regression: read-layer **153 passed**, `test_data` **49 passed**.

### 2026-06-07 (10) — pushdown works through Select/Subset (real cutouts) + crash fix

Real-cutout benchmarking exposed two things:
1. **Crash**: a cutout whose constituents are wrapped (real cross-source cutouts use
   per-member `select`/`adjust` → `Select`/`Subset`) crashed on a grid-subset shard
   — the grid array hit `Select.collect_read_parts → index_to_slices` raising
   `ValueError`, which the gate (catches only `NotImplementedError`) didn't handle.
2. Even without the crash, pushdown never engaged for real cutouts (always wrapped).

Fixes:
- `Cutout._pushdown_supported` (cached, I/O-free probe): if any constituent can't
  take a grid-axis index array, pushdown is disabled → eager full read. No crash.
- Made `Select` (var axis) and `Subset` (date axis) **grid-array transparent**:
  peel the last-axis index array (`split_grid_index`, moved to `indexing.py`), do
  the var/date transform, reattach, delegate. Pushdown now reaches the leaf through
  them.
- Tests: `TestPushdownThroughWrappers`. Read-layer **151 passed**, `test_data` **49**.

Real benchmark (`metno-meps` 2.5km Norway LAM 1.0M pts in `n320` globe 535k, 85
vars, output 1.55M) via `grid_pushdown_benchmark.py cutout --select-common
--no-var-check`:
- 1/16 globe-region shard **3.43× faster, 4× less mem** (skips the 1M LAM store).
- 1/16 LAM-region shard **2.41× faster, ~3× less mem** (skips the 535k globe).
- spanning parity (falls back); FULL time parity but **+48% memory** in two-step
  (buffer holds both parts + concat; eager frees sooner — possible future tidy).
See `benchmarks/RESULTS.md`.

### 2026-06-06 (5) — D2 union factorization (shared store read once under grid-subset)

Completed the last grid-pushdown piece. `factorize` now unions grid indices of
parts that read the **same store** with the same non-date slices:
- grouping key for grid-index parts dropped the specific `grid_index` →
  `(id(data), slices[1:], "grid")`; non-grid parts keep `(…, None)` (unchanged).
- within a grid group: read the **sorted union** of all needed points once
  (`merged.grid_index = tuple(union)`), and map each original part to its
  columns via `grid_cols = [position[g] for g in p.grid_index]`.
- `ReadBuffer.__getitem__` mapping is now a 3-tuple `(merged, row_offset, grid_cols)`;
  it does `data[offset:offset+rows]` then, if `grid_cols`, `data[..., grid_cols]`.

Effect: a `multi=` of cutouts sharing a global, each asking a *different* grid
subset, reads the global **once** (was once-per-cutout). Non-grid reads keep the
exact previous date-bounding-box behaviour.

Tests: `tests/test_grid_pushdown.py::TestMultiCutoutGridUnion` (two cutouts share a
globe + grid subset → exactly one globe part, byte-identical per member);
`test_read_parts.py::test_read_buffer_grid_cols` (cache column selection). Updated
`test_read_buffer_merged_offset` to the 3-tuple mapping.

Regression: read-layer **148 passed, 1 skipped**; `test_data.py` **49 passed**.

### 2026-06-06 (4) — A4 kill-switch + D grid-subset pushdown (the big I/O win)

**A4 — env var is now a real kill-switch.** `ANEMOI_DATASETS_READ_PARTS` defaults
on; `=0`/`false`/`no` forces the eager path. `READ_PARTS_ENABLED` default flipped
to True; the gate (`Dataset.__init_subclass__`) reads `read_parts.READ_PARTS_ENABLED`
each call so it's runtime-toggleable (tests use it to get a eager oracle result).

**B — oracle already exists.** No new framework needed: `test_read_parts_phase2`
(two-step == eager across cutout/concat/join/merge/givenaxis/…), `phase3` (gate
consistency + fallback), `phase4` (always-on + complex-wrapper fallback), `phase5`
(parallel), `verify` (missing-date checks, list/array fallback, C-fallbacks). This
is the safety net D was gated on. Marked B covered.

**D — grid-subset pushdown implemented.** When a cutout/store is indexed with a
grid *subset*, only the needed grid points are read (zarr orthogonal indexing)
instead of the full grid.
- `usage/read_parts.py`: `ReadPart.grid_index` (optional int tuple on last axis);
  `execute` → `data.oindex[..., grid_index]`; `factorize` groups by
  `(id(data), slices[1:], grid_index)`; identity/repr updated.
- `usage/store.py`: `split_grid_index` peels a grid-axis int array off a full
  tuple index; leaf `collect_read_parts`/`read_from_buffer` emit a `grid_index` part.
- `usage/gridded/grids.py` (Cutout): `_pushdown_plan` (full-LAM + masked-globe
  output layout), `_grid_pushdown`, `_is_full_grid_slice`; `collect_read_parts`/
  `read_from_buffer` push `index[3]` down. Guarded to non-full slices, step ≥ 1,
  consistent single/non-overlapping-LAM; else falls back to the eager full read.
- Tests `tests/test_grid_pushdown.py` (4): byte-identity vs eager across slice
  shapes (within LAM / across boundary / within globe / single point / strided),
  and reads **exactly the requested number of grid points**; plain-store grid-list
  pushdown; int-date+grid-list (two-step handles it, eager can't).

Regression: read-layer suites **146 passed, 1 skipped**; `test_data.py` (real
cutout/grids/join/adjust) **49 passed**. phase2 `TestCutout` (slices/tuples) now
runs through pushdown and stays byte-identical.

**Remaining (D2 — union factorization):** different `grid_index` on the same
store are not merged, so `multi=` + grid-subset combined reads a shared global
once *per cutout* (correct, but the multi win degrades). Unioning grid indices /
needed chunks per store is the next optimisation. Full-grid `multi=` still shares.

### 2026-06-06 (3) — thorough review: no regressions, corner cases checked

Reviewed the `multi=` change for bugs / lost functionality / corner cases.

Regression runs (all via `rtk proxy … -o addopts=""`):
- read-layer suites (test_read_parts ×6 + multi + indexing + datasets + parts):
  **142 passed, 1 skipped**.
- `test_data.py` (the big usage suite — join/concat/cutout/grids/ensemble/adjust/
  subset/select): **49 passed**.
- Full suite **collects** with no import errors (2956 tests).
- `test_trajectories.py`: 73 passed, **29 errors that are PRE-EXISTING** — the
  fixture `make_trajectories_zarr` calls zarr-v3 `root.create_array` but the env
  has zarr 2.18.7. Proved by `git stash` of my edits → identical errors. Not mine.

Corner cases checked:
- **repeat_dates** — it's a *create-side source* (`create/sources/repeated_dates.py`),
  not a read-layer wrapper. My changes are read-layer only ⇒ unaffected.
- **join + adjust** — untouched: the `multi` branch was added *after* `join` in
  `_open_dataset`; `_auto_adjust` is unchanged; `from_name_or_path` is a no-op
  outside `shared_zarr_opens`. `test_data.py` exercises join/adjust → passes.
- **trajectories** — `from_name_or_path` (where the share-cache lives) dispatches
  to `TrajectoriesZarr` unchanged; a `multi` of trajectory stores would share too.
- **member options** (`select`/`start`/… inside a member spec) — opened via
  `_open(spec)` per member; share-cache only affects leaf opens, wrappers are
  per-member, so subset/select on a shared leaf is correct.

Fixes/improvements made during review:
- Added `Multi.collect_supporting_arrays` + `Multi.metadata_specific` overrides so
  cutout masks/metadata are namespaced under the member name (the inherited
  `Combined` versions warn and mis-key). Important for cutout-in-multi training.
  Verified: masks collect as `('a','lam_0')`, `('a','global')`.
- Added tests: supporting-arrays namespacing + length-mismatch rejection.
  `tests/test_multi.py` now **8 passed**.

Confirmed purely additive: `shared_zarr_opens` defaults off (contextvar None),
`from_name_or_path` behaviour is byte-identical when the cache is inactive, and
the only `_open_dataset` change is a new `multi` branch. No functionality removed.

### 2026-06-06 (2) — implemented `multi=` and confirmed it helps multiple cutouts

Implemented the `multi=` container end-to-end and **confirmed the multiple-cutout
benefit the user asked about**.

Code:
- `usage/store.py`: `shared_zarr_opens()` context manager + a contextvar
  open-cache; `ZarrStore.from_name_or_path` returns the *same* `ZarrStore` for
  repeated opens of a name while the block is active. This is what makes sharing
  real — without it, `ReadPart` identity (`id(self.data)`) differs per open and
  `factorize` can't dedup.
- `usage/gridded/multi.py`: new `Multi(Combined)` + `multi_factory`. `Multi[n]`
  returns `{name: array}`; implements `collect_read_parts` (gather all members)
  and `read_from_buffer` (dict) so the gate runs one shared factorize/execute pass.
  Falls back per-member if a member is non-rectangular. Metadata exposed as
  name-keyed dicts; members must share length, grids may differ.
- `usage/misc.py::_open_dataset`: added the `multi` branch (sibling of `cutout`).

Tests (`tests/test_multi.py`, 6 passed):
- correctness: `multi[n]` == standalone members, two-step == eager, single member.
- **usefulness (the confirmation):** two cutouts (different LAMs) over one shared
  global ⇒ `factorize` → **3 reads (lamA, lamB, globe), not 4**; exactly one
  merged part is the globe. Contrast test: separately-opened globes ⇒ 4 reads.
  Generalises to **N+1 vs 2N** for N cutouts — the expensive global is read once.
- end-to-end `open_dataset(multi=dict(a={cutout:[lamA,globe]}, b={cutout:[lamB,globe]}))`
  on on-disk stores: factory's `shared_zarr_opens` makes it 3 reads; results match
  standalone cutouts.

Regression: `test_read_parts_phase2/phase3` still 52 passed. ADR "Problem 2"
section rewritten from proposal → "IMPLEMENTED" with the measured benefit.

**Verdict on the user's question ("is this actually useful for multiple
cutouts?"): yes, demonstrably** — N cutouts sharing a global drop from 2N to N+1
physical reads of the constituent stores, and the saving is on the *expensive*
full-resolution global store. (Note: this is read-count/dedup of whole-grid reads;
the *further* win of not over-reading the grid when sharding is workstream D,
still open and the bigger prize.)

Note: pytest in this venv needs `-o addopts=""` via `rtk proxy` to run a single
file (pyproject sets `--numprocesses=auto`, and the rtk hook summarises pytest
output otherwise).

### 2026-06-06 — initial analysis + ADR refresh
- Read the implemented two-step code across `usage/` and compared to
  `adr-3-two-step-read.md`. Found the divergences above.
- **Deep-think conclusion (the user's "old code is still needed" question):** the
  eager recursive `__getitem__` is **permanent**, not a migration leftover. It is
  the *only* path for non-rectangular wrappers (interpolation, rolling average,
  fill/skip missing, zip/chain, complement, tabular) and the correctness oracle
  for the rest. A casual read suggests it's deletable because the common
  open→select→subset→concat→cutout→train path is fully two-step; it is not.
- Updated `adr-3-two-step-read.md`: status → "Accepted, partially implemented";
  added "Implementation as built", "Why the eager `__getitem__` path is
  permanent", "Next steps / open design: cutout grid-subset + `multi=`"; revised
  migration Phase 4 (do not remove old path) and Phase 5 (threads done).
- Created `GOAL.md`, `PLAN.md`, this `progress.md`.

### 2026-06-06 (8) — the ~35% cutout full-read regression is FIXED

Root-caused via interleaved decomposition (one process, kills S3 noise):
- `execute_parts(threads=2)` = 590 ms; `execute_parts(threads=1)` = 415 ms ≈
  eager 443 ms; single globe read ≈ 400 ms either way.
⇒ culprit is the **cross-part `ThreadPoolExecutor`**: reading the 112 MB
single-chunk globe + LAM chunk concurrently contends on the GIL during blosc
**decompress** (CPU-bound, not I/O), so 2 threads were ~45% slower than sequential.
Within-array S3 chunk fetch is already parallel in zarr/anemoi `getitems`, so the
pool added contention with no benefit.

Fix (final): **removed the cross-part `ThreadPoolExecutor` entirely** —
`execute_parts(parts)` is now sequential, no `num_threads` param, no
`READ_PARTS_THREADS` global / env var. (First tried flipping the default 2→1, but
that exposed a smell: `num_threads=READ_PARTS_THREADS` was a frozen default-arg, so
runtime mutation of the global was a no-op — and the phase5 tests that patched it
were testing nothing. Cleaner to delete the speculative parallelism.) The
concurrency that helps — S3 latency overlap across one array's chunks — already
lives in zarr/anemoi `getitems`. Verified: full cutout read **0.97×** eager (was
0.74×); LAM-shard pushdown unchanged (**26×**); `multi=` globe-sharing preserved
(dedup, not threading). Rewrote `test_read_parts_phase5.py` around the sequential
contract. Regression suite: read-layer **149 passed**, `test_data.py` **49 passed**.

**Tension from entry (7) is RESOLVED** — the fix is the thread default, so we keep
two-step-as-default (multi sharing intact) AND parity on full reads. No need to
route cutout full reads back to eager.

### 2026-06-06 (7) — pushdown guard + a real-data regression to investigate

**Guard implemented** (`Cutout._grid_pushdown`): only push down when the plan
**skips ≥1 constituent store** (`len(plan) < len(segments)`); otherwise return
None → the two-step full-read path. On single-chunk-per-field stores there's no
within-chunk saving, so a shard touching every store gains nothing and the
`oindex` gather is slower than a contiguous slice. A globe-only shard still skips
the LAM, so it keeps pushing down. Tests updated (spanning shards fall back;
multi-union test uses a globe-region shard). Read-layer suite **150 passed**.

**BUT benchmarking surfaced a bigger, pre-existing issue.** Real cutout
(`o48` in `n320`), interleaved timing (no ordering bias; min values confirm):
- LAM-region 1/16 shard: two-step **31× faster** (skips the 112 MB globe). ✓
- **FULL-grid cutout read: two-step ~0.74× (≈35% SLOWER) than eager `_get_tuple`.**
  Same for spanning shards → the guard does NOT speed up spanning; the cost is the
  two-step *framework*, not the oindex gather it removes.

Diagnosis (not fully root-caused; S3 latency variance confounds):
- A **plain store** full read is fine (~1.0×) → slowdown is **cutout-specific**.
- **Same chunk count and bytes** fetched both ways → not I/O volume.
- Extra time is in **`execute_parts`** (the read), not reconstruction.
- **Not** the explicit-slice form; **not** thread count (1/2/4 same).
- Leading suspect: per-read `ThreadPoolExecutor` in `execute_parts` interacting
  with anemoi's own parallel chunk `getitems` for the large single-chunk globe.

**Tension:** making cutout full/spanning reads fall back to eager `_get_tuple`
(raise NotImplementedError) would fix the ~35% — but disables `multi=` full-read
globe sharing (workstream C, a real N≥2 win). Needs the cutout to know whether
sharing applies (e.g. Multi opts members into two-step; standalone cutout uses
eager for non-pushdown reads). Deferred — not a one-liner.

### Benchmark finding (2026-06-06 (6)) — REAL S3 data; corrected understanding
Access: `open_dataset('<name>')` (by name) resolves to `s3://ml-datasets/...` with
working creds — that's why it succeeds where the root-owned FUSE `/s3` path and
direct `zarr.open` failed. Use names, never the FUSE path.

Chunking survey (15 production datasets, o48→o1280 globals + LAMs to 3M pts):
**all keep the whole grid in ONE chunk** (`data.chunks[-1] == grid`). Matches the
anemoi default (`get_chunking` → grid axis gets full length). ⇒ **no within-chunk
byte saving exists on current data.**

Real cutout benchmark (`o48` in `n320`, one chunk each, 1/16 grid shard, 101 vars):
- shard in LAM (globe store skipped): eager 443ms/234MB → pushdown **17ms/6MB
  (25×/42×)**.
- shard in globe (one-chunk store must be read): 408/234 → 371/220 (~1.1×).
- shard spanning boundary (both read via oindex gather): 403/234 → 531/220
  (**0.8× — slightly slower**).

**Corrected conclusion** (supersedes the earlier synthetic "100%/14%" framing):
the win is **skipping constituent stores the shard doesn't touch** (+ not
materialising the full-grid concat → memory win on real big-globe cutouts), NOT
within-chunk trimming. Boundary-spanning shards are a minor (~20%) regression —
candidate for a future guard ("only push down if it skips ≥1 store"), but the
memory win for big-globe cutouts argues against a blanket guard. Full-grid reads
unaffected. Kill-switch is the escape hatch.

### Benchmark committed (2026-06-06 (9))
Reproducible benchmark + recorded results live in
`docs/adr/two-step-read/benchmarks/` (`grid_pushdown_benchmark.py` with
`survey`/`cutout`/`synthetic` modes; `RESULTS.md` has the `ewc-vm-ewc-s3` run:
EWC VM, datasets on EWC S3). Headline real numbers: full cutout read 1.02× eager
(parity), 1/16 LAM-region shard **25× faster / ~46× less memory** (skips the
112 MB globe store), globe/spanning shards ~parity.

### Next actions (as of 2026-06-06 (9))
Workstreams A–D functionally complete; benchmark committed. Remaining polish:
- A3 (minor): explicit named guard test for a eager-only wrapper via fallback
  (behaviour already covered by phase4 / verify).
- Public docs for `multi=` + grid-subset (only in the ADR today; both experimental).
- Optional: a byte-saving win needs the grid axis chunked at dataset creation
  (`values` chunk in the recipe) — data-eng change, out of scope for the reader.
