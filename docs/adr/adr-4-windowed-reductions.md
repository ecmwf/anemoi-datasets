# Temporal reductions beyond accumulate: two loops by kind of data, one validity table

## Status

Proposed - 14/10/2026

## TL;DR:

> Today `accumulate:` is the only source that reduces a time window when a dataset is built. It can only sum, and it never checks what the archived fields mean.
>
> This ADR adds `maximum:`, `minimum:` and `average:` as recipe blocks beside `accumulate:`. All four read both instantaneous and interval-valued source data. **One table** decides whether a reduction suits what the archive stores.
>
> The code is split by **what the source data is**, not by **what operation is applied**: two plain loops, four reduction functions and one table. A generic plan/state abstraction serving both loops was built, reviewed and rejected. Readability won over abstraction.

## Context

### What exists today

`accumulate:` produces one output field per date by **summing** the window `(date − period, date]` of an archive of accumulated fields:

- the window is covered by archived intervals (`covering:` / `availability:`);
- each subwindow is rebuilt by a **signed sum**: one field when the archive holds it outright, `+a(0,12) − a(0,6)` when it accumulates from the start of the run;
- the output is stamped `proc.time_method: accum`.

That is the only temporal reduction available when a dataset is built. The only other is `rolling_average`, used when a dataset is opened: it takes a mean over a fixed number of the dataset's own time steps, and only for values already in the dataset. Nothing produces a window maximum, minimum or mean from the archive.

### What users ask for

| need | field | stored as |
|---|---|---|
| maximum wind gust in the past 6 h | `10fg` | `[0-1] [1-2] [2-3] …`; CERRA: `[0-1] [1-2] [2-3] [3-6] [6-9]` |
| minimum 2 m temperature in the past 6 h | `mn2t` | `[0-1] [1-2] [2-3] …` |
| mean short-wave flux over 6 h | `avg_sdswrf` | `[0-1] [1-2] [2-3] …` (means over each interval) |
| total precipitation in the past 6 h | `tp` | hourly increments, or cumulative from the run start |
| mean short-wave flux from a stored integral | `ssrd` | cumulative from the run start |
| wettest hour in the past 6 h | `tp` | hourly increments, or cumulative from the run start |
| monthly mean of daily maximum temperature | `mx2t24` | one 24 h interval per day |
| daily mean / maximum of analyses | `2t` | instantaneous, every 1–6 h |

Only the fourth row works today, and some of the others are worse than unsupported: `accumulate:` over a gust archive *builds*, and writes meaningless values (see below).

### Two kinds of source data, and what each field means

The difference between kinds of source data matters more than the difference between operations:

- **Instantaneous** fields exist every `frequency` (6-hourly analyses). A window is a list of **samples**.
- **Interval-valued** fields each *span* an interval: a gust archive stores the maximum since the previous field, and a precipitation archive stores a total. A window is cut into **subwindows**, each rebuilt from archived fields by a signed sum. This is the model `accumulate:` already uses.

Each archived field also carries a **statistic** over its own interval: `accum`, `max`, `min`, `avg`, or `instant`. GRIB2 states it in `proc.time_method`. GRIB1 often says `instant` whatever the field is, and only the parameter knows (`stepTypeForConversion`).

`accumulate:` never reads this statistic. Over a gust archive it returns `max(0,9) − max(0,6)` and writes it out as an accumulation.

### Which reductions make sense

The key property is **partition invariance**: whether the answer depends on how the window was cut into pieces.
- A sum is the same however the window was cut, and so is the maximum of stored maxima.
- A mean of stored maxima is not: 3-hourly maxima average higher than hourly ones.

**Rows** are the recipe block, which is the reduction asked for. **Columns** are the statistic each archived field carries. The verdicts describe `main` today:

| | ✅ | ⚠️ | ⭕ | ❌ |
|---|---|---|---|---|
| | makes sense, works | makes sense, works only by workaround or on trust | makes sense, not possible | does not make sense |

| block / stored statistic | accumulations | maxima | minima | means / rates | instantaneous | unstated |
|---|---|---|---|---|---|---|
| `accumulate:` | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| `maximum:` | ⭕ | ⭕ | ❌ | ⭕ | ⭕ | ⭕ |
| `minimum:` | ⭕ | ❌ | ⭕ | ⭕ | ⭕ | ⭕ |
| `average:` | ⚠️ | ⭕ | ⭕ | ⭕ | ⚠️ | ⭕ |

What the cells mean:

- **✅ `accumulate:` over accumulations** — a sum is partition-invariant. Every layout works today: hourly increments, cumulative from the run start, reset cycles, lookup tables.
- **❌ `accumulate:` over maxima, minima or means** — adding stored statistics gives no physical quantity. `main` computes it anyway, without a warning.
- **❌ `accumulate:` over instantaneous fields** — a sum of samples depends on how often you sampled. It cannot be refused, though (see *unstated*).
- **❌ crossed extrema** (the maximum of stored minima) — arithmetically defined, physically meaningless.
- **⭕ extremum of extrema** (`10fg`, `mn2t`) — partition-invariant, and the core need.
- **⭕ extremum of accumulations** (the wettest hour) — needs each subwindow rebuilt by a signed sum *and then* reduced, plus a key stating how long a subwindow is, because the largest hourly total and the largest 3-hourly total are different numbers.
- **⭕ mean of extrema** (monthly mean of daily Tmax, an ETCCDI index) and **extremum of means** (peak hourly flux) — real quantities, but partition-dependent. They are well defined when the archive's own subwindows *are* the definition: every subwindow held outright and all the same length.
- **⭕ mean of means** (`avg_sdswrf`) — partition-invariant only when the mean is **weighted by length**.
- **⭕ reductions of instantaneous fields** (daily mean or maximum of analyses) — the simplest case, sampled every `frequency`.
- **⚠️ mean of accumulations** — equal to the window total divided by `period`. Possible today only as `accumulate:` followed by scaling outside the recipe block, and the output is then stamped `accum` while holding a rate.
- **⚠️ mean of instantaneous fields** — only through `rolling_average` when the dataset is opened: over the dataset's own steps, and never stored.
- **⚠️ unstated** — accepted on trust. GRIB1 reports genuine accumulations as `instant` (`od` `cp`), so refusing `instant` would reject most of the archives `accumulate:` exists to read.

### What this ADR has to settle

- Adding `maximum:`, `minimum:` and `average:` turns the ⭕ cells into working reductions. To do that they must read both kinds of source data, and a window must be modelled at two levels: subwindows rebuilt by a signed sum, then reduced.
- Making every cell trustworthy needs the archived statistic checked against the reduction. That covers `accumulate:` too.

The question is how to structure the code that does this.

This builds on two pieces developed on `feature/hindcasts`:
- the `from:` key, which describes what the source data is and replaces `covering:` / `availability:`;
- the move to earthkit-data 1.x.

It is gated by the recorded MARS requests per test recipe (`tests/create/requests/`, #774): every existing recipe must keep asking MARS for exactly the same fields.

## High-level principles

- **Code readability above abstraction.** A reader following one field from the subsource to the output should not have to cross more layers than there are concepts.
- **Validity rules live in one place.** What may be reduced by what is a table, not a property scattered across classes.
- **Fail loudly.** A reduction that would silently be wrong is refused, with the reason and what to write instead. A window missing a field is an error, never a mean over what happened to arrive.
- **No change in what is retrieved.** Existing `accumulate:` recipes keep making the same MARS requests.
- **The reduction is the name of the block.** `maximum:`, `minimum:` and `average:` sit beside `accumulate:`. There is no `reduce:` source and no `operation:` key.

## Options considered

Both options share the same description layer: `from:`, the covering search, and `Subwindow`, which is one piece of the window plus the archived intervals whose signed sum rebuilds it. They differ in how the loops and the rules are organised.

### Option A — a generic plan (rejected)

One base class, `WindowSourceBase`, runs one loop for every windowed source. What differs between kinds of data is pushed into a `WindowPlan` interface with two implementations:

- `IntervalPlan` for interval-valued data;
- `SamplingPlan` for instantaneous data.

Each window holds one `State` per piece:
- `SubwindowState` rebuilds its piece by a signed sum;
- `SampleState` takes one field as it is.

A `Reducer` combines the states with an `Operation` (`Sum`, `Max`, `Min`, `Mean`).

```
WindowSourceBase._reduce_fields
  └─ plan.identify / field_info / candidates / new_reducer / offer / note / unused_field
       └─ Reducer.compute
            └─ State.accept / release
                 └─ Operation.reduce / finalize
```

This was implemented in full and worked. What it cost:

- **Indirection.** `WindowPlan` has nine hooks and only two implementations. Following one field means reading five files (`source.py` → `plan.py` → `interval_plan.py` → `reducer.py` → `states.py`).
- **Validity rules in six places**, checked at two moments:

  | where | when |
  |---|---|
  | `Operation.reduces_archived` | every field × every window |
  | `Operation.differenceable` | when the windows are planned |
  | `NON_ADDITIVE` + `State.differences` | every field × window × piece |
  | `IntervalPlan._check_reducible` | when the windows are planned, per window |
  | `IntervalPlan._check_not_degenerate` | when the windows are planned, per window |
  | `SamplingPlan` | never |

  The last row was a real gap. The instantaneous path refused nothing, so it computed a mean of stored maxima that the interval path refused.
- **Size.** 2281 lines across the plan, state, reducer, operation and source modules.

### Option B — direct loops split by kind of data (chosen)

There is no base class, plan interface or state layer. Each kind of data has its own plain loop:

- **Interval-valued data** uses the `accumulate:` loop with an operation parameter (`windowed/ranges.py`, `windowed/reducer.py`). Each subwindow is a signed sum of archived fields. When a subwindow is complete, its value is reduced into the window value by one of four functions: sum, max, min, or length-weighted mean (`windowed/operations.py`).
- **Instantaneous data** uses a sampling loop (`reduce/source.py`): a reverse index from sample time to windows, and one `SampleReducer` per window.
- **Shared helpers** are plain functions (`windowed/common.py`): setting up the subsource, the grouping key, and the completeness check.
- **Validity** is one function, `check(operation, statistic, pieces)` (`windowed/check.py`). It is driven by a dictionary `TABLE[(operation, statistic)]`, and both loops call it once per variable, when that variable's first field arrives.

```
accumulate / maximum / minimum / average
  ├─ interval-valued: windowed/ranges.py    reduce_ranges ───┐
  └─ instantaneous:   reduce/source.py      _reduce_samples ─┤
                                                             ├─ check.check (once per variable)
                                                             └─ operations.OPERATIONS[...]
```

### Comparison

| | Option A — plan | Option B — direct |
|---|---|---|
| layers a field crosses | 4 (source → plan → reducer → state) | 2 (loop → reducer) |
| places validity is decided | 6, at two moments | 1, at one moment |
| instantaneous path checked | no | yes |
| lines (loops, reducers, operations, validity) | 2281 | 1656 |
| loop code | one generic loop | two loops, about 150 lines each |

Option A avoids having two loops. That is its one advantage, and the two loops are short and genuinely different: matching samples by validity time is not matching intervals by their endpoints. Two simple loops are easier to read than one generic loop plus a plan interface.

## Decision

**Option B.** `maximum:`, `minimum:` and `average:` are added beside `accumulate:`. The windowed sources are split by kind of source data, with one validity table.

### The recipe

The three new blocks take the same keys as `accumulate:`: `source:` for where the data comes from, `period:` for the window wanted, and `from:` for what the source data is. `from:` is required, because a reduction cannot be recognised from the source the way an accumulation can.

```yaml
# instantaneous source data, sampled every 6 h
average:
  period: 24h
  from: {frequency: 6h}
  source: {mars: {class: ea, type: an, levtype: sfc, param: [2t]}}

# interval-valued source data, described as accumulate: describes it
maximum:
  period: 6h
  from:
    base_dates: {times: [0, 12]}
    steps: ["0-1", "1-2", "2-3", "3-6", "6-9", "9-12"]
  source: {mars: {class: rr, type: fc, levtype: sfc, param: [10fg]}}

# the wettest hour of a cumulative archive
maximum:
  period: 24h
  over: 1h
  from:
    base_dates: {times: [0]}
    steps: {start: 1h, end: 48h, frequency: 1h}
    accumulation: from-zero
  source: {mars: {class: od, type: fc, levtype: sfc, param: [tp]}}
```

The window is end-anchored and half-open, `(date − period, date]`, as for `accumulate:`. The output is stamped with `proc.time_method` `max`, `min` or `avg` over `period`.

#### A rolling window on instantaneous data

There is no `rolling:` key. A window is rolling when `period` is longer than the dataset's output frequency, so consecutive windows overlap. A rolling 24 h mean over 6-hourly data:

```yaml
dates:
  start: 2020-01-01T00:00:00
  end: 2020-01-31T18:00:00
  frequency: 6h          # the output stride: one field every 6 h

input:
  average:
    period: 24h          # longer than the stride, so the windows overlap
    from: {frequency: 6h}
    source: {mars: {class: ea, type: an, levtype: sfc, param: [2t]}}
```

The field stamped `2020-01-02T12:00` is the mean of the four samples at `−18 h`, `−12 h`, `−6 h` and `0 h`, that is `01-01T18:00`, `01-02T00:00`, `01-02T06:00` and `01-02T12:00`. The start of the window belongs to the previous one. The next field, at `18:00`, reuses three of the four samples, and each source field is fetched once and offered to every window that needs it.

- `from: {frequency: 6h}` must divide `period`, so the window is a whole number of samples.
- The first windows reach back before `start`, so the source must hold samples from `start − period + frequency`. A window missing a sample raises, instead of averaging fewer samples.
- Under `maximum:` or `minimum:` the recipe is the same with the block renamed, for a rolling 24 h maximum.
- This differs from `over:`, which cuts one window into fixed subwindows of interval-valued data. Here the windows themselves slide, by the dataset's stride.
- It is the build-time counterpart of `rolling_average`, which averages the dataset's own time steps when a dataset is opened.

### The validity table

```
block \ stored   accum          max      min      avg            instant / unstated
accumulate:      ok             ✗        ✗        ✗              ok
maximum:         ≥2 equal (*)   ok       ✗        D+U            ok
minimum:         ≥2 equal (*)   ✗        ok       D+U            ok
average:         ✗ (hint)       D+U      D+U      ok (weighted)  ok

D+U  = every piece held outright, all pieces the same length
(*)  = at least two pieces per window, all the same length; `over:` makes them so
max / min / avg columns: any piece rebuilt from several fields is refused (not additive)
```

The cell decisions:

1. **Partition-dependent cells are accepted only when the archive's own subwindows are the definition** (D+U): every piece held outright and all the same length. This keeps the monthly mean of daily Tmax and the peak hourly mean flux. It refuses the same request on an archive whose granularity changes with lead time, where the result would mix hourly and 3-hourly subwindows.
2. **`average:` over accumulations is refused**, with the message "use `accumulate:` and divide by the period". A first-class mean rate needs a parameter and units change (J m⁻² → W m⁻²), which is left to a follow-up.
3. **Both loops are checked.** On the instantaneous loop every sample is a piece held outright, `frequency` long.
4. **A mean accumulated from the start of the run is refused.** Rebuilding one would need `m(6,9) = (9·m(0,9) − 6·m(0,6)) / 3`, which is not implemented, so a differenced `avg` piece fails loudly. Checked: `ea` `avg_sdswrf` is a mean over each step interval, and `od` `oper` does not archive it.
5. **`instant` and unstated statistics are accepted everywhere, on trust.**
6. **The check runs once per variable, when its first field arrives**, because that is when the statistic is known. Where a field says nothing and `from:` declares an `accumulation`, the description is used instead. It runs per variable, not once per build, because one request can mix `tp` and `10fg`. It is given the pieces of every window, because one archive can hold one window outright and need the next differenced (`od` runs are 12 h apart, windows 6 h).

### `over:`

- `over:` is a new key for interval-valued data. It cuts the window into **fixed, consecutive** subwindows, each rebuilt by a signed sum of the fields covering it. So `over: 1h` on a cumulative archive gives hourly totals, and `over: 3h` on an hourly archive gives 3-hourly totals.
- `over:` applies only to additive fields; on stored maxima, minima or means it is refused.
- The subwindows are fixed, not rolling: `[0-3] [3-6]`, not every 3 h at an hourly stride. The two agree when `over:` equals the archive's own step.
- `over:` is recorded only in the stored recipe, not in the field metadata.
- `accumulate:` takes no `over:`, because a sum does not depend on the partition.

### Trajectory layout: the run from the layout

Under `layout: trajectories` every output row is a `(base_date, step)` pair from one forecast run. What the data source is asked for depends on `from:`, not on the layout. The one case that differs is **the run from the layout**, where the source data is the row's own run:

| block | `from:` | the source is asked for |
|---|---|---|
| `maximum` / `minimum` / `average` | `{frequency: ...}` | validity dates; the row's base date only stamps the output |
| `maximum` / `minimum` / `average` | `{base_dates: true, frequency: ...}` | lead times of the row's run (`ForecastDates`) |
| all four | `{base_dates: from-layout, steps: from-layout, accumulation: ...}` | step ranges of the row's run (`ForecastIntervals`) |
| all four | any other interval-valued `from:` | intervals found by searching the archive; the row's base date only stamps the output |

For example, the 6 h maximum gust at step 12 of the 00Z run reads steps `[6-7]` … `[11-12]` of the 00Z run, and nothing from any other run:

```yaml
maximum:
  period: 6h
  from:
    base_dates: from-layout
    steps: from-layout
    accumulation: 1h        # each field holds the maximum over one step
  source: {mars: {class: od, type: fc, stream: oper, levtype: sfc, param: [10fg]}}
```

- **Two argument types, because of the kind of data.** A sample is a point: `(valid_time, basetime)` identifies it, so `ForecastDates` is enough. A field that spans an interval is not: at step 12 of the 00Z run an archive may hold `[11-12]`, `[9-12]` or `[0-12]`, and a window needs several such fields, some subtracted. `ForecastIntervals` therefore carries the output rows and, separately, the archived intervals the covering found.
- **One definition.** `takes_run_from_layout(from_)` is true for both spellings, and both recipe rules follow from it: only valid under `layout: trajectories`, and `steps.start ≥ period`, so that the window lies inside the run.
- **No search.** The window is covered by `ForecastCovering`, anchored on the row's run, exactly as `accumulate:` does. The validity table then applies unchanged: hourly gust maxima are pieces held outright, so `maximum:` is accepted; a from-zero description over maxima would need differenced pieces, and is refused.
- **Data sources.** Only sources that can fetch a field by run and lead time serve it: `mars`, `grib`, `grib-index`, and `forcings` for samples. A source that cannot address a field by the interval it spans now raises a clear error. Before this change the base `Source` silently fell back to asking for the output dates only.

### Interval-valued fields described as instants

A recipe may also describe interval-valued fields as if they were instants (`from: {frequency: 1h}` over hourly gusts). The sampler keys on validity time, so this gives the right answer wherever the table allows it, and the same table guards it. It also reaches MARS forecast archives:
- `_expand_mars_request` expands each validity time by the recipe's `step:` list;
- a wildcard `date: "????-??-??"` plus a `time:` filter keeps exactly one run per validity time.

It is supported but not advertised; the interval-valued `from:` is the documented spelling.

## Scope of Change

`anemoi-datasets`, `create` only. No change to the dataset layout or to the usage API.

- New sources `maximum`, `minimum` and `average` (`sources/reduce/`).
- New package `sources/windowed/` for what all four windowed sources share:
  - moved from `sources/accumulate/`: covering, search, interval generators, field-to-interval, clipping, `from:` descriptions;
  - new: `subwindows.py`, `operations.py`, `check.py`, `common.py`, `ranges.py`, `reducer.py`.
- `sources/accumulate/source.py` runs on the interval loop with `sum`.
- Recipe:
  - three new blocks, and the key `over:`;
  - `from:` replaces `covering:` / `availability:`, which remain accepted with a deprecation warning.
- `Source.execute_intervals` / `execute_forecast_intervals` no longer fall back to the validity-date handlers; a source without interval support raises `NotImplementedError` naming the sources that work.
- earthkit-data has no `min` time method, so it is registered at import time until it is available upstream.
- Docs: `docs/building/sources/average.rst`, including the validity table and the semantics of `over:`.

## Consequences

- Window maxima, minima and means can be produced when a dataset is built: from analyses, from per-step archives (including irregular ones), and, with `over:`, from cumulative archives (the wettest hour). Under a trajectory layout they can be taken from each row's own forecast run.
- **Behaviour change for data sources:** a source that cannot serve archived intervals now fails at once with an explicit error, instead of being asked for the output dates and failing later with missing fields.
- One table states, in one place, what may be reduced by what. The user documentation reproduces it, and adding a statistic or an operation means adding a row or a column.
- Following one field through the code means reading one loop and one reducer.
- **Behaviour change for `accumulate:`:** over an archive of stored maxima, minima or means it now stops with an explanation, instead of writing meaningless values. Recipes that relied on that output fail. That output was never correct.
- Existing `accumulate:` recipes keep making the same MARS requests.
- **Trade-off:** there are two loops rather than one. They share helpers, operations and the table, but a change to the shape of the loop (logging, diagnostics) has to be made twice.
- **Deferred:**
  - CERRA / `rr` stamps `10fg` with `startStep=0` whatever range it covers, so its fields do not match declared step pairs; a possible opt-in patch would rebuild the start step from `from:`;
  - rolling subwindows (`stride:`);
  - recording `over` in the field metadata;
  - a first-class mean rate;
  - removing the `min` registration once earthkit-data provides it.

## Alternatives Considered

- **Option A, the generic plan** — rejected for the reasons above. Its data structures survive: `Subwindow` and the partition invariants (`validate_contributions`, `validate_partition`).
- **Reductions when the dataset is opened**, extending `rolling_average` — it only sees values already in a dataset, counted in dataset steps. It cannot rebuild a subwindow from archived intervals, does not know what the archive stored, and stores nothing.
- **Refusing every partition-dependent combination** — simplest table, but it drops quantities people use, such as the monthly mean of daily Tmax.
- **Accepting every combination** — produces archive-dependent numbers without warning on any archive whose granularity changes with lead time.
- **A `reduce:` source with an `operation:` key** — rejected: the reduction is the verb, so it is the name of the block, and `accumulate:` stays as the additive spelling.

## References

- #774 — recorded MARS requests per test recipe (`tests/create/requests/`).
- `feature/hindcasts` — the `from:` description and the move to earthkit-data 1.x.
- Branches `refactor/reductions` (Option A) and `refactor/reductions-split` (Option B).
- `docs/building/sources/average.rst` — the user-facing table and the semantics of `over:`.
