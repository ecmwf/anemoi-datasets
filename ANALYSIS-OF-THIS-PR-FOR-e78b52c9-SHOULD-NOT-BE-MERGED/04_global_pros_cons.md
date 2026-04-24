# 04 — Global architecture: pros and cons

Opinionated analysis of the architectural choices the branch makes. One
bullet per choice, each with a **Pro**, a **Con**, and a pointer to the
code where the tradeoff manifests. The point is *not* to say the branch is
wrong — most of these are reasonable tradeoffs. The point is to expose
them so downstream decisions can be informed.

## 1. Typed argument hierarchy with MRO fallback

> `ValidDates` / `ForecastDates` / `Intervals (⊂ ValidDates)` /
> `ForecastIntervals (⊂ ForecastDates)`.

**Pro.** A source that only wants "a list of snapshots per window" can
register only `@for_valid_dates` and still receive `Intervals` via MRO
lookup. This is a genuine polymorphism win: the 2×2 matrix does not
multiply the boilerplate a simple source has to write.

**Pro.** The type is the documentation. Reading a source class is enough
to know what kinds of recipes can target it; no need to cross‑reference an
`if isinstance(dates, X)` ladder.

**Con.** The inheritance is subtle. `Intervals` has `.dates` *and*
`.intervals` — if a `@for_valid_dates` overload treats it as a plain
`ValidDates` it loses the interval information silently (by design). A
reader who does not know this will read `Intervals` as a list of valid
times and be surprised when a sibling source produces per‑window data.

**Con.** The conversion surface (`as_intervals`, `with_basetime`,
`as_forecast_intervals`, `adjust_request`) lives on the arguments
themselves, so the arguments are both value objects *and* request
rewriters. `adjust_request` mutates nothing but builds a MARS‑specific
request dict, which is a responsibility that arguably belongs on the
source or a helper. Code: `create/arguments.py:203-234, 307-330`.

## 2. Frame‑inspection `@for_*` dispatch

> `_MultiDispatch` descriptor built via `sys._getframe(1).f_locals`.

**Pro.** The recipe author's mental model — "one source, one `execute`"
— is preserved. All the dispatch magic is one import.

**Pro.** Zero‑runtime cost after construction: a single dict lookup in
the descriptor's `__get__`, plus an MRO walk only on miss.

**Con.** `sys._getframe(1)` is brittle. Metaclasses, `exec`,
class‑body decorators that wrap further, unusual import hooks — any of
these can break the accumulation trick. The docstring itself
([`create/dispatch.py:44-48`](../src/anemoi/datasets/create/dispatch.py#L44-L48))
warns about `LegacySource` needing to precede `DispatchedSource` in MRO.

**Con.** Static analysis can't see it. Mypy will flag the two `def
execute` definitions as redefinitions; IDE go‑to‑definition jumps to the
last one. The `@overload` pattern would cost more boilerplate but would
be legible to tools.

**Con.** Error messages mention the *last* registered overloads, not the
call site. A user who passes the wrong type gets
`"'X' has no overload for argument type 'Y'"` but no hint of which
recipe tree node triggered it. Code:
[`create/dispatch.py:122-126`](../src/anemoi/datasets/create/dispatch.py#L122-L126).

## 3. `SignedInterval` with optional `base`

> `base=None` means the interval came from a valid‑time‑indexed backend.

**Pro.** One class covers both the forecast case (basetime imposed) and
the archive case (basetime discovered by the covering search). The
algebra of `+`/`−` intervals falls out of `sign` and `__neg__`.

**Con.** The `base is None` / `base is not None` asymmetry leaks into
every consumer. Three overloads (`@for_intervals` in MARS, FDB,
grib‑index) carry explicit `assert` statements about `.base`, and
`Intervals.adjust_request` has to assert and raise for base‑less cases
because grib‑index does not go through it. The invariant lives in
comments and asserts rather than in the type system — it could be two
types (`ValidTimeInterval` without `base`, `ForecastInterval` with
`base`) at no runtime cost.

## 4. Context owns `order_by`, not the recipe

> `SimpleGriddedContext.order_by` and `TrajectoryGriddedContext.order_by`
> are class attributes; `output.order_by` is deprecated.

**Pro.** The cube ordering is *tightly coupled* to the layout (the
trajectories case injects a composite `traj_point` remapping key that the
gridded case doesn't have). Forcing the layout to own it means invalid
combinations don't even compile.

**Pro.** Migration is painless: old recipes with the hard‑coded default
keep working (warning only); recipes that actually relied on a different
order fail with a clear message.

**Con.** A user who needs a different ordering — say, moving
`number` before `param_level` for an ensemble‑first workload — now has to
patch the framework. Previously they could change the recipe.

**Con.** The `output.order_by` → context move leaves the recipe model
with a deprecated field that exists only for back‑compat; the Pydantic
discriminator sees it, validates it, then discards it. Code:
[`recipe/output.py:73-99`](../src/anemoi/datasets/create/recipe/output.py#L73-L99).

## 5. `dates` vs `base_dates` on the store

> Trajectories Zarr stores `base_dates` and `steps`; gridded stores
> `dates`. Readers alias `dates` to `base_dates` for back‑compat.

**Pro.** Each layout's Zarr naming reflects what the array actually
means. A reader opening the store sees `base_dates` and `steps` and
immediately knows it's not a plain gridded dataset.

**Pro.** Shared statistics / loading code keeps working because
`Dataset.dates` falls back to `base_dates`
([create/dataset.py:217-224](../src/anemoi/datasets/create/dataset.py#L217-L224)).

**Con.** The alias *hides* the semantic difference. `dataset.dates[0]`
on a gridded store is the first validity time; on a trajectory store it
is the first *base* date — subtly different in graphs, statistics, and
filtering. The `start_date` / `end_date` attributes encode the envelope
for trajectories, which is yet another convention; a reader must know
both to interpret a mixed dataset collection.

**Con.** `Dataset.frequency` is an explicit `AttributeError` for
trajectories
([usage/trajectories/store.py:248-257](../src/anemoi/datasets/usage/trajectories/store.py#L248-L257))
with a hint. Any code that accesses `.frequency` unconditionally — and
there is a lot of it in the wider anemoi ecosystem — now branches on
layout or try/except.

## 6. `Covering` vs `Availability` split

> `IntervalGenerator` describes the archive; `Covering` picks intervals
> to cover a window. Previously they were fused inside
> `SearchableIntervalGenerator`.

**Pro.** The forecast case needed a trivially different picker
(`ForecastCovering`), and the split made it a 90‑line class instead of a
whole new branch in Dijkstra. The archive case is a pure wrapper
(`AutoCovering`) around the existing search.

**Pro.** The `covering_factory` discriminator form is future‑proof
(`auto` / `cycle` / future strategies) and legacy‑compatible.

**Con.** The deprecation spreads across four files: the recipe source
(`accumulate/source.py:87-93`), the command‑line migrator
(`commands/recipe/migrate.py:190-210`), the factory
(`accumulate/covering.py:198-254`), and the tests
(`tests/create/test_covering_factory.py`). Removing the old key will
require touching all four.

**Con.** `ForecastCovering` is not reachable from the recipe. The
`{forecast: …}` discriminator raises explicitly to say "this branch is
selected implicitly by the argument type". That is correct behaviour,
but it introduces a concept — *implicit dispatch by argument type* —
that is not obvious to a recipe author who only sees `covering:` and
`accumulation:`.

## 7. Creator × Result × Context per layout

> `SimpleGriddedCreator`/`TrajectoryGriddedCreator`,
> `SimpleGriddedResult`/`TrajectoryGriddedResult`,
> `SimpleGriddedContext`/`TrajectoryGriddedContext`.

**Pro.** Shared logic lives on the abstract bases (`GriddedCreator`,
`GriddedResult`). Each layout's quirks (tuple dates, 5‑D array, 4‑D
array, composite remapping key) are isolated to one file.

**Pro.** `_metadata_dates` / `_metadata_date_range` hooks avoid
`if is_trajectory` branches in the shared code.

**Con.** Three places need to move together for any new layout. Adding,
say, a "horizontal ensemble" layout means a new Creator *and* a new
Result *and* a new Context, plus a new discriminator branch in
`creator.py` and an Output subclass in `recipe/output.py`. The pattern
is clear but it is a 5‑point change per layout.

**Con.** `collect_metadata` is the only shared `super().collect_metadata(...)`
chain; the trajectory override has its own overrides for `layout`,
`steps`, `ensemble_dimension`, `step_dimension`, and base‑date envelope.
There is no test that a future layout gets the right ordering of
`super().collect_metadata()` → subclass overrides.

## 8. `TrajectoryGroups` yields `ForecastDates`, not `GroupOfDates`

> Other `Groups` yield `GroupOfDates`; trajectory groups break the
> contract to avoid retyping at every source entry.

**Pro.** The typed argument arrives at the source verbatim; no
back‑compat wrapping needed for the common path.

**Con.** `GroupOfDates` and `ForecastDates` are isinstance‑tested in
`Result.__init__`
([gridded/result.py:333-335](../src/anemoi/datasets/create/gridded/result.py#L333-L335))
and in the dispatch closure
([create/dispatch.py:99-108](../src/anemoi/datasets/create/dispatch.py#L99-L108)).
A third yield type would need the same two updates, easy to miss.

**Con.** `GrouperByKey._key` has to introspect tuples
([dates/groups.py:332-337](../src/anemoi/datasets/dates/groups.py#L332-L337))
so the common `group_by: monthly` works for trajectory providers — a
small but real cost to the `Grouper` abstraction.

## 9. Cube `order_by` keys always end in `(variables, ensembles)`

> The `build_coords` implementation reads the last two keys as
> `(variables, ensembles)` and hands any leading keys to
> `_post_build_coords`.

**Pro.** Exactly one hook for any new layout to add extra coordinate
axes (trajectories uses it for `step`).

**Con.** The assumption is not typed. A future layout that decides
`ensembles` comes before `variables` would silently flip them without an
assertion. Code:
[`gridded/result.py:586-597`](../src/anemoi/datasets/create/gridded/result.py#L586-L597).

## 10. Flattening always on

> `flatten_values=True` hard‑coded;
> `flatten_grid` removed from recipe and Zarr attrs.

**Pro.** One fewer degree of freedom in the Zarr layout means one fewer
thing to check when a downstream tool is opening someone else's store.

**Con.** Trajectories explicitly coupled itself to flat cells (`grid_values
= list(range(n_cells))`); a future layout wanting non‑flat cells would
need to walk back through `build_coords` and the creator
`initialise_dataset`. There is no test that guards against reintroducing
`flatten_grid` as a recipe key.

## 11. Migration and deprecation policy

> `availability:` and `output.order_by` both deprecated in one release;
> migrate tool rewrites `availability:`; `output.order_by` does not have a
> command‑line migrator.

**Pro.** Migration tool entry point (`_fix_accumulate_availability`)
exists and is tested
([tests/create/test_covering_factory.py:47-89](../tests/create/test_covering_factory.py#L47-L89)).

**Pro.** `user_date` / `user_time` (undocumented but real) now raise a
clear `ValueError` pointing to the wildcard shorthand
([sources/mars/retrieval.py:198-204](../src/anemoi/datasets/create/sources/mars/retrieval.py#L198-L204)).

**Con.** `output.order_by` has no migrator. Users will see a
`DeprecationWarning` — fine — but there is no `anemoi datasets recipe
migrate` step that strips the key. The only way to know is to read the
deprecation message.

**Con.** `output.flatten_grid` was removed without deprecation. Old
recipes carrying it would fail at Pydantic validation (extra key) unless
`OutputBase` is configured with `extra="ignore"`. The branch does not
spell this out.

## 12. New reader‑side wrappers (`StepSubset`, `SingleStepView`,
`Subset`)

**Pro.** Open‑dataset semantics for `step=`, `steps=`, `step_start=`,
`step_end=`, `step_frequency=`, `base_start=`, `base_end=` are clean and
composable with the existing `start=`/`end=`/`frequency=` machinery.

**Pro.** `SingleStepView` yields a 4‑D array that is
shape‑indistinguishable from a gridded dataset — so downstream code that
already handles gridded can consume a single forecast step without a
special case.

**Con.** `Forwards` has to forward every trajectory‑specific property
(`base_dates`, `base_start_date`, `base_end_date`, `base_frequency`,
`step_frequency`, `start_date`, `end_date`) by hand
([usage/forwards.py:97+](../src/anemoi/datasets/usage/forwards.py#L97)).
A new trajectory‑specific property means one extra line on every
forwarder, and there is no base class that enforces this.

**Con.** The `open_dataset` kwarg surface is now larger — 7 new kwargs
— and they interact with the existing envelope logic (`_dates_to_indices`
uses a stricter rule for trajectories).

## 13. Hindcasts and recentre skip dispatch

> Both `HindcastsSource` and `RecentreSource` still inherit from
> `LegacySource` and call `fire_prebuilt_requests` /
> `execute_mars_request` directly.

**Pro.** They don't need the typed arguments: hindcasts already has the
date‑mapping logic in `HindcastsDates`, and recentre composes two MARS
requests rather than dispatching on argument type.

**Con.** They will not work in a trajectory recipe. A recipe author who
reads `@for_forecast_dates` on `MarsSource` and assumes any MARS‑like
source can live under `base_dates:` will find that `hindcasts:` and
`recentre:` don't route. Today this is an implicit constraint enforced
only by runtime errors.

**Con.** The split "some sources dispatch, some don't" means the rule
"every source under a trajectory recipe must handle `ForecastDates`" is
documented by failure mode, not by a type.

## 14. Composite / Pipe ABC exists but is unused

> `create/composite.py` introduces `Composite` ABC and `Pipe` helper.

**Pro.** It's one file, very small, and gives future refactors a place
to stand. The separation from `anemoi.transform.filter.Filter` is
spelt out in the docstring.

**Con.** Today no source uses it. The risk is that it becomes the kind
of abstraction that sits unused, drifts, and is eventually seen as
legacy — cheap to add now, expensive to remove.

## 15. Summary — direction of travel

The branch consistently moves *configuration* out of the recipe and into
the code: cube `order_by`, `flatten_grid`, and the `forecast` branch of
`covering:` are no longer user knobs. The *interface*
(`ValidDates`/`ForecastDates`/`Intervals`/`ForecastIntervals`) becomes
the single vocabulary between the creator and the sources. The tradeoff
across the board is the same: fewer ways to hold the framework wrong, at
the cost of a larger mental model and a richer set of runtime assertions
when an edge case slips through.

## Pointers

- `create/arguments.py` — argument hierarchy.
- `create/dispatch.py` — `_MultiDispatch`.
- `create/intervals.py` — `SignedInterval`.
- `create/recipe/{__init__,output}.py` — recipe validation.
- `create/gridded/{creator,context,result}.py`,
  `create/trajectories/{creator,context,result}.py` — layout triples.
- `create/sources/accumulate/covering.py` — covering layer.
- `dates/{__init__,groups}.py` — providers and groups.
- `usage/trajectories/{store,subset}.py`, `usage/forwards.py`,
  `usage/dataset.py` — reader side.
- `tests/test_trajectories.py`, `tests/create/test_covering_factory.py`,
  `tests/create/test_forecast_covering.py`,
  `tests/test_request_filter.py` — tests that pin the design.
