# Fresh review — two-step read branch (soundness audit)

Date 2026-06-07. Reviewer: fresh pass over the whole two-step read implementation
(committed phases 1–2 + this session's grid-pushdown / `multi=` / rename / guards).

## Goal

Independently verify **decisions** and **implementation** are sound. Not "do tests
pass" (they do) — but: is each design choice defensible, is each code path correct
including edge cases, is anything subtly wrong, over-claimed, or fragile?

## Plan (checklist)

- [x] R1. Scope. 20 files (873+/241-), + new `multi.py`, `grid_pushdown` tests.
- [x] R2. Core `ReadPart`/`factorize`/`ReadBuffer`/`execute_parts` — sound.
- [x] R3. Gate + kill-switch — sound; fallback is `NotImplementedError`-typed (nit).
- [x] R4. Leaf — sound; tuple int-date OOB modulo-wraps (nit, pre-existing, verified).
- [x] R5. Select/Subset grid-transparency + `Forwards.chunks` — sound.
- [x] R6. Cutout pushdown — sound (mirror, order, identity round-trip verified).
- [x] R7. Multi + shared_zarr_opens + factory — sound; grid-index across
  heterogeneous members has per-member semantics (nit).
- [x] R8. Decisions — all defensible; claims honest after earlier corrections.
- [x] R9. Tests — phase2–5 + verify + grid_pushdown + multi cover the claims;
  byte-identity oracle present. 164 read-layer tests green.
- [x] R10. Cross-cutting — sequential (no thread issue); **+48% memory full cutout
  read** is the one material concern; docs accurate post-review.

## Recommendations (priority order)
1. 🟠 Decide on the **+48% memory full-cutout-read** cost (drop consumed buffer
   entries, or route full-grid cutout reads to eager with `multi=` carve-out).
   Biggest real downside; everything else is cosmetic. (Still open.)
2. ✅ ~~tuple int-date bounds check~~ — DONE (`check_int_bounds`).
3. ✅ ~~fallback contract~~ — DONE (`None`-return, not exception).
4. ✅ ~~A3 guard test~~ — DONE + an **oracle** (`test_two_step_oracle.py`) proving
   two-step never silently diverges from eager across all dataset types × edge
   indices. Corner-case hunt found **zero** further divergences.

## Findings (severity: 🔴 bug / 🟠 risk / 🟡 nit / 🟢 ok)

### Decisions — all defensible
- 🟢 **Eager path permanent.** Non-rectangular ops (interp / rolling / fill /
  zip / chain / complement / tabular) cannot be `ReadPart`s. Correct, well-argued.
- 🟢 **Sequential `execute_parts` (pool removed).** Evidence-based: blosc decompress
  is GIL-bound; within-array S3 fetch already parallel in `getitems`. Measured.
- 🟢 **Grid pushdown** — claims now honest: on 1-chunk grids (all real data) the win
  is *skipping whole stores* + smaller output (memory), measured 2.4–3.4× on real
  cutout shards; within-chunk byte saving only if grid is chunked (none exist yet).
- 🟢 **Guard conditioned on `_grid_is_chunked`.** Correct, forward-looking.
- 🟢 **`multi=`** — sound for its purpose (shared global read once). Niche but real.
- 🟢 **`_pushdown_supported` probe** — sound safety net; redundant with Select/Subset
  transparency (belt-and-suspenders), acceptable.

### Implementation
- 🟢 **`factorize` grid-union math** (R2). Group `(id(data), slices[1:], "grid")`;
  union sorted, `position` map, `grid_cols` preserve per-part order incl
  duplicates; date-bbox + grid axes independent → reconstruction correct. Non-grid
  path unchanged.
- 🟢 **`ReadBuffer.__getitem__`** — date offset then `[..., grid_cols]`, independent
  axes. Correct.
- 🟢 **Cutout collect ↔ read_from_buffer mirror** (R6). Both call identical
  `_is_full_grid_slice` + `_grid_pushdown` (deterministic, cached plan) → same
  branch. Segment order = ascending output order = eager `result[..., slice]`
  order. `grid_index` tuple is deterministic across both calls → `ReadPart`
  identity round-trips → buffer hit. Squeeze applied once at Cutout level.
- 🟢 **`total != self.shape[-1]` guard** disables pushdown for multi-LAM-overlap
  (where eager `shape` already disagrees with `_get_tuple`). Conservative, correct.
- 🟢 **Select/Subset grid-transparency** — grid array peeled (`split_grid_index`),
  transform applied to other axes, reattached; mirror in read_from_buffer. Correct.
- 🟢 **Gate** (`__init_subclass__`) — per-class wrap, no double-wrap, reads
  `READ_PARTS_ENABLED` each call (patchable), catches `NotImplementedError`→eager.

### Risks / nits to track
- 🟠 **FULL cutout read uses +48% memory under two-step** (the DEFAULT path; training
  reads full samples). `execute_parts` holds every constituent array in the buffer
  AND `read_from_buffer` builds the concat → both alive at peak (measured 1830 vs
  1238 MB on meps→n320). Cutout-specific (plain store ≈ parity). For large cutouts
  near memory limits this is the most material downside of two-step-as-default.
  Mitigations (not done): drop buffer entries once consumed; or route full-grid
  cutout reads to eager (but that disables `multi=` full-read sharing — tension).
- ✅ **FIXED — Fallback now by return value, not exception.** `collect_read_parts`
  returns ``None`` to request eager; `gather_parts` propagates a child ``None``;
  gate uses `None`→eager. The 14 unsupported wrappers `return None` (no raise).
  A narrow `(ValueError, TypeError, AttributeError)` catch in `two_step_read`
  remains only for non-normalisable fancy indices. Tests updated (verify/phase3/4).
- ✅ **FIXED — Tuple int-date out-of-bounds now raises.** Leaf `check_int_bounds`
  validates int indices (bare or in a tuple) against shape *before* `index_to_slices`
  → `IndexError`, matching eager (was silently `i % size`-wrapping → date 0).
  Verified both paths; test `test_out_of_bounds_int_date_raises_both_paths`.
- ✅ **RESOLVED — `multi` per-member indexing.** Added `ds[{name: index}]`: each
  member indexed by its own index (the explicit way to grid-shard a multi; I/O
  still shared, D2 union). Broadcast (`ds[t]`) kept for shared-date reads, and the
  per-member semantics are now documented on `Multi`. (Was: broadcasting one grid
  slice to heterogeneous-grid members — surprising.)
- 🟡 **`_pushdown_supported` / `_grid_is_chunked` use broad `except`.** If a probe
  hits e.g. `MissingDateError` on date 0 of a constituent, pushdown silently
  disables (falls back — correct, just unoptimised). Acceptable.

### Verdict
Core math and the collect↔read mirror are **sound**; no correctness bug found in
the new code. One material non-correctness concern: the **+48% memory on full
cutout reads** under the always-on two-step default. Everything else is nits /
documentation. Tests + oracle (phase2–5, verify, grid_pushdown, multi) cover the
claims; byte-identity vs eager is asserted across the supported wrappers.
