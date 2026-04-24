# 05 — Per‑source pros and cons

One section per source. Each fine‑grained observation has a **Pro**, a
**Con**, and a file:line pointer. The intent is to highlight *specific
implementation choices inside each source*, not the architectural themes
covered in `04_global_pros_cons.md`.

## 1. MarsSource

### 1.1 Package split: `source.py` vs `retrieval.py`

**Pro.** `hindcasts.py`, `recentre.py`, `from_trajectories.py`, and
`fdb.py` now import their MARS primitives from one module and never
touch the source class. The stable surface is explicit
([sources/mars/__init__.py](../src/anemoi/datasets/create/sources/mars/__init__.py)).

**Con.** The public surface is still implicit — the `__init__.py`
re‑export is the only affordance that marks these as API. There is no
test that guards import‑graph stability.

### 1.2 Four `@for_*` overloads in one class

**Pro.** Every MARS use case (valid dates, forecast dates, archive
intervals, forecast intervals) is visible on one screen at
[sources/mars/source.py:46-144](../src/anemoi/datasets/create/sources/mars/source.py#L46-L144).
Easy to audit.

**Con.** Each overload rebuilds `base_requests = list(self.args) or
[self.kwargs]` — four copies of the same idiom. `_reject_filters` does
not run for `@for_valid_dates`, which is where filters *are* valid.
Easy to add a new overload and forget whichever of the four pre‑amble
steps it needs.

### 1.3 Empty‑dates path for repeat‑dates constant mode

**Pro.** Without this path, `DateMapperConstant(date=None)` — the only
way to attach a fixed‑request source to a broadcast — would not work at
all. Code:
[sources/mars/source.py:57-66](../src/anemoi/datasets/create/sources/mars/source.py#L57-L66).

**Con.** It is detected by `if not dates.dates:`, i.e. by the empty list
content, not by a dedicated type. A `ValidDates([])` is conceptually
different from `ValidDates([t0, t1])` — the former is "please fire the
prebuilt request", the latter is "expand for each date". A separate
argument type (`ConstantDates`?) would make the distinction explicit;
the branch doesn't do that.

### 1.4 `RequestFilter`

**Pro.** Rejects `user_date` / `user_time` with a loud `ValueError`
pointing to the new shorthand — a genuine UX upgrade on silent success.
Code:
[sources/mars/retrieval.py:198-204](../src/anemoi/datasets/create/sources/mars/retrieval.py#L198-L204).

**Pro.** `_compile_date` handles `int`, `datetime`, `str` inputs; the
regex rewrite (`?` → `.`) is clean.

**Con.** The filter only exists in the validity‑date expansion path.
The forecast/interval/forecast‑interval paths refuse filters entirely
(`_reject_filters`); as a consequence, a user who wants to restrict base
dates in a trajectory recipe has no equivalent shorthand — they must
write the `from-trajectories` wrapper with `bases: "????-..."`. The two
mechanisms do not share code.

### 1.5 `stream: scda` auto‑selection

**Pro.** Silent‑but‑correct fix for the ECMWF operational idiom (06/18
UTC runs live under `scda`). Code:
[sources/mars/retrieval.py:306-308](../src/anemoi/datasets/create/sources/mars/retrieval.py#L306-L308).

**Con.** Only applied in `_expand_mars_request` (valid‑date path). The
other three paths skip the rewrite. If a trajectory recipe asks for 06Z
basetime data via `@for_forecast_dates`, the source will send `stream:
oper` and the retrieval will fail quietly. The TODO at
[sources/accumulate/source.py:39-45](../src/anemoi/datasets/create/sources/accumulate/source.py#L39-L45)
flags this.

### 1.6 `MARS_KEYS` whitelist

**Pro.** Rejects unknown keys with `did_you_mean` suggestions. Code:
[sources/mars/retrieval.py:359-363](../src/anemoi/datasets/create/sources/mars/retrieval.py#L359-L363).

**Con.** Every new MARS key requires editing the list. The list is
~70 items today; there is no automated sync with MARS documentation.

## 2. FdbSource

### 2.1 Only two overloads (valid + intervals)

**Pro.** The registered overloads cover the two cases today's users have
asked for (archive analysis + accumulations).

**Con.** No `@for_forecast_dates` means trajectory recipes cannot use
FDB. Cross‑reference: `MarsSource` has four overloads, and the TODO
comment in both files says "code duplication between here and MARS
might be reduced". The asymmetry is an unfinished migration, not a
design decision.

### 2.2 Importing `compress_prebuilt_requests` from the MARS package

**Pro.** The FDB source is now *decoupled* from the accumulate package
(which used to re‑export a different primitive). Import graph is
smaller.

**Con.** The rename (`factorise_requests` → `compress_prebuilt_requests`
for already‑expanded requests) drops the factorising of the validity
dates. That's correct for `@for_valid_dates` — each date becomes one
request — but means the FDB path cannot merge identical `(date, time)`
tuples the way MARS does via `Availability.iterate()`. In practice FDB
has its own cache; in principle this is a regression for some use cases.
Code:
[sources/fdb.py:87-94](../src/anemoi/datasets/create/sources/fdb.py#L87-L94).

### 2.3 Flavour + grid hooks shared across overloads

**Pro.** `_execute_requests` is the single chokepoint, so both branches
share post‑processing. Code:
[sources/fdb.py:111-134](../src/anemoi/datasets/create/sources/fdb.py#L111-L134).

**Con.** The MARS source does not have the same chokepoint — flavour /
grid is not relevant there — so the two sources now diverge on *where*
post‑processing can be added. A future "rename parameters" hook would
need to go in two places.

## 3. GribIndexSource

### 3.1 Base‑less intervals as a first‑class case

**Pro.** Keeps the abstraction honest: `SignedInterval.base=None` is not
a workaround, it is the genuine case for a valid‑time‑indexed archive.
The assertion at
[sources/grib_index.py:635-638](../src/anemoi/datasets/create/sources/grib_index.py#L635-L638)
enforces the invariant.

**Con.** The helper `Intervals.adjust_request` *requires* `base is not
None` and grib‑index has to bypass it, so the "shared helper" is shared
by two of three consumers. Duplicated stamping logic in
[sources/grib_index.py:640-643](../src/anemoi/datasets/create/sources/grib_index.py#L640-L643).

### 3.2 Own `factorise()` for requests

**Pro.** Grib‑index does not need MARS `Availability` semantics — just a
simple `(dates, request)` dedup on request hash. One function, 15 lines,
near the source. Code:
[sources/grib_index.py:669-684](../src/anemoi/datasets/create/sources/grib_index.py#L669-L684).

**Con.** Two `factorise`‑flavour functions in the codebase
(`mars/retrieval.factorise_requests` vs `grib_index.factorise`). Naming
is close, semantics diverge. A contributor tracking "where does request
dedup happen?" will have to check both.

## 4. AccumulateSource

### 4.1 Shared pipeline `_accumulate_fields` for both branches

**Pro.** `_accumulate_fields` treats the archive and forecast branches
uniformly by parameterising on `(valid_date, basetime)` targets
(basetime=None for archive). No `if self.is_forecast:` branch inside the
accumulation loop. Code:
[sources/accumulate/source.py:166-230](../src/anemoi/datasets/create/sources/accumulate/source.py#L166-L230).

**Con.** The dispatch between `write_accumulated_field_with_valid_time`
and `write_accumulated_forecast_field` lives inside `Accumulator`
instead, which means the *writer* is picked at accumulator‑construction
time via the presence of `basetime`. It works, but the rule is not
obvious from the caller's side. Code:
[sources/accumulate/accumulator.py:121-137](../src/anemoi/datasets/create/sources/accumulate/accumulator.py#L121-L137).

### 4.2 `covering` legacy back‑compat

**Pro.** Four code paths accept the old `availability:` key cleanly:
- constructor kwarg (`source.py:87-93`);
- `covering_factory` legacy branch (`covering.py:253-254`);
- recipe migrator (`commands/recipe/migrate.py:190-210`);
- test (`tests/create/test_covering_factory.py`).

**Con.** The same four code paths all need updating when we eventually
remove the alias. A simple deprecation cycle requires a coordinated
change.

### 4.3 `covering: { forecast: … }` explicitly rejected

**Pro.** The forecast branch is driven by the argument type, not the
recipe. The explicit rejection with a pointer to `accumulation:` is
exactly the right error message for a user who reads about
"covering" and expects the forecast variant to be a discriminator key.
Code:
[sources/accumulate/covering.py:243-249](../src/anemoi/datasets/create/sources/accumulate/covering.py#L243-L249).

**Con.** The "forecast branch is implicit" rule is one more concept a
user has to know. It makes the recipe‑level `accumulate:` block
asymmetric (has `accumulation` for forecast, has `covering` for
archive). A user reading the YAML cannot tell which one their recipe
will activate without checking the surrounding `base_dates:` / `dates:`.

### 4.4 Hash‑caching the inner source object

**Pro.** Same `(period, source)` across multiple `accumulate:` blocks
(e.g. same MARS request, three different periods) will collapse to the
same cached source. The forecast branch includes `accumulation` in the
hash so `from-zero` and `from-previous-step` aren't collapsed. Code:
[sources/accumulate/source.py:122-127](../src/anemoi/datasets/create/sources/accumulate/source.py#L122-L127).

**Con.** MD5(json.dumps(…, sort_keys=True)) as a cache key is brittle —
anything that doesn't round‑trip through `json.dumps` silently forks the
cache. The input is a dict from YAML, so in practice this is fine, but
there is no guard.

### 4.5 `from-zero` / `from-previous-step` as a flag

**Pro.** One flag, two branches, an `_VALID_ACCUMULATIONS` tuple, clear
error messages. Code:
[sources/accumulate/covering.py:101-134](../src/anemoi/datasets/create/sources/accumulate/covering.py#L101-L134).

**Con.** Only relevant on the forecast branch (archive search picks the
scheme implicitly from the availability generator). The caller has to
*not* set `accumulation` on an archive recipe — today that is silent.
A warning‑or‑error for `accumulation` on the archive branch would match
the symmetry of the `covering` branch.

### 4.6 `patch_groupby_keys` defaults

**Pro.** Default `{"namespace": "mars", "ignore": ["date", "time",
"step"]}` is what every MARS accumulator wants; overrides only for the
rare case. Code:
[sources/accumulate/source.py:48-60](../src/anemoi/datasets/create/sources/accumulate/source.py#L48-L60).

**Con.** `namespace != "mars"` raises a hard error ("use 'mars'"),
which is too strict for a future FDB‑based accumulate. The assumption
that all accumulators live in the MARS namespace is encoded here.

## 5. HindcastsSource

### 5.1 Bypassing dispatch entirely

**Pro.** Hindcasts have their own date‑mapping provider and a pre‑built
request list per hindcast — dispatch on argument type would be overhead,
not simplification.

**Con.** Without a `@for_*` overload, `HindcastsSource` cannot live under
a trajectory recipe. The recipe grammar silently allows `hindcasts:` in
any recipe, but it will raise at runtime with a type mismatch if the
argument is a `ForecastDates`. A registration refusal at source factory
time would be cheaper.

### 5.2 Date format change (`%Y-%m-%d` → `%Y%m%d`)

**Pro.** Aligns with the MARS `date` convention already used by every
other source; removes a class of accidental mismatches when the same
request dict is shared across sources. Code:
[sources/hindcasts.py:79-82](../src/anemoi/datasets/create/sources/hindcasts.py#L79-L82).

**Con.** Old recipes that relied on `hindcasts` producing the dashed
format (unlikely but possible) will fail silently against downstream
readers. The change has no test coverage beyond the integration test
`tests/create/hindcasts.yaml`.

## 6. RecentreSource

### 6.1 One‑line import switch

**Pro.** `from .mars.retrieval import execute_mars_request` keeps the
source class untouched — all the plumbing is in the MARS package.

**Con.** Recentre still bypasses dispatch (same critique as Hindcasts).
It works only on `ValidDates`‑equivalent input.

## 7. RepeatedDates / DateMapperConstant

### 7.1 `date: null` supported

**Pro.** Honest encoding of "this field does not depend on any date".
Previously users had to write dummy dates and hope the source ignored
them. Code:
[input/repeated_dates.py:246-259](../src/anemoi/datasets/create/input/repeated_dates.py#L246-L259).

**Con.** The contract "source must handle an empty date list" is now
load‑bearing. Only `MarsSource` and `FdbSource` handle it explicitly;
other sources (`grib_index`, `accumulate`, …) would fail with an
unrelated error. A test at the input‑tree level that asserts "empty
date list reaches the source" would pin this.

### 7.2 Eager parsing of `date:`

**Pro.** Invalid values fail at build time, not at fetch time. Code:
[input/repeated_dates.py:253-259](../src/anemoi/datasets/create/input/repeated_dates.py#L253-L259).

**Con.** `as_datetime` accepts a wide range of formats — including some
that parse but are clearly not what the user meant (e.g. `"today"`).
There is no opinion on which formats should be rejected.

## 8. FromTrajectoriesSource

### 8.1 Not a `DispatchedSource`

**Pro.** The source *transforms* its argument into the forecast form
and then calls `self.inner.execute(forecast_dates)`. Dispatching the
outer source would add noise: the inner source is the one with the
typed‑argument overloads.

**Con.** It means a gridded `input:` block containing
`from-trajectories:` has the odd property that *one* of its sources is
un‑dispatched. Documentation will need to explain that
`from-trajectories:` is a wrapper rather than a leaf source.

### 8.2 fnmatch pattern on basetime strings

**Pro.** Fairly natural for users who already know wildcards (`?`, `*`).
Reuses a stdlib primitive. Code:
[sources/from_trajectories.py:101-111](../src/anemoi/datasets/create/sources/from_trajectories.py#L101-L111).

**Con.** Different semantics from the MARS wildcard shorthand
(`????-??-01` in `RequestFilter`) — which is a regex at heart. The two
wildcards live in different layers (user‑facing in from‑trajectories;
MARS internal in `RequestFilter`) but a user who uses both will see
subtly different behaviours on edge cases.

### 8.3 Smallest step first

**Pro.** If two steps produce valid basetimes, the shorter lead time
wins. Defensible default for most use cases.

**Con.** Not configurable. Some users might want a specific step
(e.g. always step 24); today the only way is to list just that step in
`steps:`. Code:
[sources/from_trajectories.py:77-132](../src/anemoi/datasets/create/sources/from_trajectories.py#L77-L132).

### 8.4 Inner source must be a single‑key dict

**Pro.** Clear error message at construction time; no silent loss of
keys. Code:
[sources/from_trajectories.py:91-95](../src/anemoi/datasets/create/sources/from_trajectories.py#L91-L95).

**Con.** A `join:` of multiple forecast sources would need two
`from-trajectories:` blocks rather than one. This is a deliberate
choice but worth flagging for recipe authors.

## 9. Sources that were untouched but are impacted

Not every source in the registry was modified. Those that remained on
`LegacySource` (e.g. `xarray`, `netcdf`, `accumulations` legacy, `url`,
…) continue to receive plain lists of datetimes. A trajectory recipe
that tries to route through any of them will fail at the dispatch
descriptor, because `GroupOfDates` is the only back‑compat wrapping —
a legacy‑only source on a trajectory recipe would never be called (the
upstream group is `ForecastDates`, which the dispatch closure does *not*
unwrap).

That is the intended behaviour, but it's implicit. A per‑source
opt‑out flag for trajectory recipes would make the constraint explicit
and give better error messages.

## 10. Cross‑source consistency table

| Source               | base‑less interval | forecast branch | `ValidDates([])` |
| -------------------- | ------------------ | --------------- | ---------------- |
| MarsSource           | rejects            | supported       | supported (special path) |
| FdbSource            | rejects            | *not supported* | supported (fallback to `self.request`) |
| GribIndexSource      | *requires*         | n/a             | handled (delegates empty dates list to `GribIndex.retrieve`) |
| AccumulateSource     | accepted (matches `(start,end)` only) | supported (ForecastCovering) | n/a (consumes `ValidDates`/`ForecastDates`, not empties) |
| HindcastsSource      | n/a                | n/a             | n/a (LegacySource) |
| RecentreSource       | n/a                | n/a             | n/a (LegacySource) |
| FromTrajectoriesSource | n/a              | wraps           | n/a (delegates) |

The inconsistencies (*not supported* vs *handled* vs *n/a*) are the
backlog the branch leaves for future work.

## Pointers

- `src/anemoi/datasets/create/sources/mars/source.py`,
  `.../mars/retrieval.py`.
- `src/anemoi/datasets/create/sources/fdb.py`.
- `src/anemoi/datasets/create/sources/grib_index.py`.
- `src/anemoi/datasets/create/sources/accumulate/source.py`,
  `.../accumulate/covering.py`, `.../accumulate/accumulator.py`,
  `.../accumulate/writers.py`.
- `src/anemoi/datasets/create/sources/hindcasts.py`,
  `.../recentre.py`.
- `src/anemoi/datasets/create/input/repeated_dates.py`.
- `src/anemoi/datasets/create/sources/from_trajectories.py`.
