.. _sources-average:

##################################
 average, minimum and maximum
##################################

**********
 Concepts
**********

Dataset to build versus source data
====================================

``average``, ``minimum`` and ``maximum`` reduce a window of **source data**
in time and write one field per date into the **anemoi dataset** being
built.

.. mermaid::

   flowchart LR
       s[source data] -->|average| d[anemoi dataset]

They are the non-additive counterpart of :ref:`accumulate
<sources-accumulate>`: the same recipe shape key for key — ``source:`` says
where the data comes from, ``period:`` says what you want, and ``from:``
says what the source data is — and the same source data. Where
``accumulate`` sums a window, these take its mean, minimum or maximum.

The reduction is the name of the block. There is no ``operation:`` key and
no generic ``reduce:`` source in a recipe.

The window
==========

``period:`` is the length of the window reduced into each output field. It
is **end-anchored** and half-open — ``(date − period, date]`` — which is
the convention an anemoi dataset uses throughout and the one
``accumulate`` reconstructs for a sum (see :ref:`the-anchor-convention`).

A 24 h window over 6-hourly source data is therefore the four samples at
``−18h``, ``−12h``, ``−6h`` and ``0h``: the start of the window belongs to
the previous window, so consecutive daily means do not share a sample.

``period:`` is independent of the dataset's own ``dates.frequency``. Equal
frequencies give a rolling reduction (consecutive windows overlap); a
coarse ``dates.frequency`` with a matching ``period`` gives a
non-overlapping resample, e.g. one daily maximum per day.

Source cadence versus dataset frequency
=======================================

``from.frequency:`` is the cadence of the **source data**, which is
generally not the frequency of the dataset being built: a daily dataset of
24 h means over 6-hourly analyses has ``dates.frequency: 24h`` and
``from: {frequency: 6h}``. Stating them separately is the point — the
window length, the source cadence and the anchoring are three facts, and
each is written once.

The window must be a whole number of source samples: ``from.frequency``
must divide ``period``, or the recipe is rejected.

``from.frequency:`` exists only for instantaneous source data. Fields that
each span an interval have no cadence to divide: the window is covered by
whole archived intervals instead — see `Interval-valued source data`_.

***************
 Configuration
***************

These keys configure a reduction over **instantaneous** source data —
fields that exist every ``from.frequency``. For fields that each span an
interval, see `Interval-valued source data`_, which has its own table.

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - key
     - meaning
   * - ``period``
     - **Required.** The window you want (``6h``, ``12h``, ``1d``, …),
       ending at the date the output field is stamped with.
   * - ``source``
     - **Required.** The data source, as a single-key dictionary
       (``source: {mars: {...}}``).
   * - ``from``
     - **Required.** What the source data is — here, the cadence at which a
       field exists. It must divide ``period``.

       -  ``{frequency: <cadence>}`` — **base-less**: instantaneous fields
          indexed by validity time, one every *cadence*. Works under either
          output layout.
       -  ``{base_dates: true, frequency: <cadence>}`` — the forecast run
          the trajectory layout imposes; see `Trajectories`_.
   * - ``group_by``
     - *Optional.* Which metadata keys identify a variable; same meaning
       and defaults as in :ref:`accumulate <how-to-group-by>`.

.. literalinclude:: yaml/reduce-average.yaml
   :language: yaml

``minimum`` and ``maximum`` take exactly the same keys — a daily maximum
of hourly 2 m temperature:

.. literalinclude:: yaml/reduce-maximum.yaml
   :language: yaml

*****************************
 Interval-valued source data
*****************************

``from: {frequency: ...}`` says a field *exists every* cadence. Some archives
are not like that: a gust archive stores the maximum since the previous field,
so every field **spans** an interval. Those are described exactly as
:ref:`accumulate <sources-accumulate>` describes them, and every ``from:``
shape it accepts works here too.

The same three keys, plus ``over:``:

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - key
     - meaning
   * - ``period``
     - **Required.** The window you want, exactly as above.
   * - ``source``
     - **Required.** The data source, exactly as above.
   * - ``from``
     - **Required.** What each field *spans*. Every shape ``accumulate``
       accepts works here and means the same thing:

       -  :ref:`base_dates + steps <trajectories>` — indexed by base date ×
          step (MARS-like). ``steps`` is a regular range plus an
          ``accumulation`` (``from-zero`` / a duration / reset), or an
          explicit list of ``"sA-sE"`` pairs for an irregular grid.
       -  :ref:`lookup-table <how-to-lookup-table>` — an explicit table, the
          escape hatch; the table *is* the description.
       -  :ref:`accumulation alone <valid-time>` — the *bare* form: indexed
          by validity time, ``accumulation`` being the length each field
          holds; or, under a trajectory layout, the run it imposes.

       Unlike ``accumulate``, it cannot be omitted: there is no recognition
       from the ``mars`` source here.
   * - ``over``
     - *Optional.* How long each subwindow is, where subwindows have to be
       constructed. Omitted, the window is covered in one go and the
       archive's own granularity decides. See :ref:`reduce-over`.
   * - ``group_by``
     - *Optional.* As above.

``patch:`` and ``clip:`` are ``accumulate``-only.

This is what makes an irregular archive expressible at all. ``rr/se-al-ec``
stores gusts over ``0-1, 1-2, 2-3, 3-6, 6-9`` — hourly to step 3,
three-hourly after — so there is no single ``frequency:`` to state, and the
step pairs are written out instead:

.. literalinclude:: yaml/reduce-maximum-gusts.yaml
   :language: yaml

The window is then covered by whole archived intervals rather than sampled,
and the reduction is applied across them.

.. note::

   ``rr`` itself stamps every ``10fg`` field with ``startStep=0``, whatever
   range it covers, so its fields do not match the declared pairs and the
   build stops with "Field not used". Until the start step can be rebuilt
   from the declared pairs, the example above only works on an archive that
   encodes its ranges.

.. _reduce-over:

Subwindows and ``over:``
========================

The pieces the window is cut into are **subwindows**, and each carries one
value that the reduction is applied across. Where they come from depends on
what the archive holds and on ``over:``.

Without ``over:`` the window is covered in one go and the archive's own
granularity decides:

-  an archive holding **parts of the window outright** — per-step gusts,
   hourly increments — gives one subwindow per archived field. A 6 h window
   of hourly increments is six 1 h subwindows, so ``maximum:`` is the
   wettest hour without your asking for anything.
-  an archive storing a **running total from the start of the run** holds no
   part of the window: covering it gives one subwindow, rebuilt by
   subtracting two cumulative fields — and a maximum over one subwindow is
   that subwindow, so it is refused.

With ``over:`` the window is cut into fixed, consecutive ``over``-long
slices, and each slice is one subwindow, rebuilt by a signed sum of the
fields covering it. ``over: 1h`` on a cumulative archive gives six hourly
totals; ``over: 3h`` on an hourly increment archive gives two 3-hourly
totals. That is only meaningful for additive fields, so ``over:`` over
stored maxima, minima or means is refused.

.. literalinclude:: yaml/reduce-maximum-over.yaml
   :language: yaml

The slices are **fixed, not rolling**: ``over: 3h`` on a 6 h window is
``[0-3]`` and ``[3-6]``, not every 3 h interval at an hourly stride. The two
agree when ``over:`` equals the archive's own step, which is why the wettest
*hour* is unambiguous on an hourly archive. ``over:`` is recorded only in
the recipe stored with the dataset — the output field says ``max`` over
``period``, whatever the slice length.

``accumulate`` does not take ``over:``. A sum of subwindows is the same
whatever length they are.

Which reductions an archive supports
====================================

Each piece of the window — a subwindow, or a sample — carries a statistic: a
total, a maximum, a mean over its own interval, or an instantaneous value.
The reduction is applied across those values, and the result is what you
asked for only when the two agree. One table decides, for every block:

.. list-table::
   :widths: 16 21 21 21 21
   :header-rows: 1

   * - block \\ fields store
     - totals (``accum``)
     - maxima
     - minima
     - means (``avg``)
   * - ``accumulate``
     - ✅
     - ❌
     - ❌
     - ❌
   * - ``maximum``
     - ✅ equal pieces, at least two per window
     - ✅
     - ❌
     - ✅ equal pieces held outright
   * - ``minimum``
     - ✅ equal pieces, at least two per window
     - ❌
     - ✅
     - ✅ equal pieces held outright
   * - ``average``
     - ❌ — accumulate and divide by ``period``
     - ✅ equal pieces held outright
     - ✅ equal pieces held outright
     - ✅ weighted by length

A field that does not state its statistic — an instantaneous analysis, or a
GRIB1 accumulation reported as ``instant`` — is accepted under every block,
on trust.

In every column but the first, a piece must be **held outright** by the
archive: maxima, minima and means are not additive, so a piece rebuilt by
subtracting or adding stored fields is refused.

"Equal pieces" is what makes the partition-dependent cells well defined. The
mean of daily maxima is a real quantity — the archive's daily blocks *are*
the definition — but over an archive whose granularity changes with lead
time it would mix hourly and 3-hourly maxima into one number, so it is
accepted only when every piece has the same length.

Why those are refused
---------------------

**Subtracting two cumulative maxima** does not give the maximum over the
difference of their intervals. If the extreme falls in the earlier interval,
both fields carry it and the difference is zero. A mean accumulated from the
start of the run could in principle be rebuilt —
``m(6,9) = (9·m(0,9) − 6·m(0,6)) / 3`` — but only weighted by length, which
is not implemented.

**A maximum over one piece is that piece**, so where the archive holds the
whole window as one accumulated field, the result would be the window's
total with a ``max`` label on it. ``od-oper`` runs are 12 h apart, so a 6 h
``period`` is held outright on every other row and differenced in between;
``over:`` fixes both.

**A mean of totals** is the window total divided by however many blocks the
archive holds. For a mean rate, use ``accumulate:`` and divide by ``period``.

**Crossed extrema** — the maximum of stored minima — and **sums of
statistics** have no physical meaning.

When a refusal happens
----------------------

Once per variable, when its first field arrives, because that is when the
field's own statistic is known — read from ``proc.time_method``, or from the
parameter on GRIB1. Where the field does not say and ``from:`` declares an
``accumulation``, the description is taken instead. The same check runs for
instantaneous ``from: {frequency: ...}`` data, where every sample is a piece
held outright.


***************
 Trajectories
***************

In a :ref:`trajectory recipe <layouts-trajectories>` the output rows are
``(base_date, step)`` and each one holds the reduction over
``(base_date + step − period, base_date + step]``. Two source shapes are
served, and ``from:`` says which.

Base-less source data
=====================

``from: {frequency: ...}`` — the same block as in a gridded recipe. The
samples are analyses, fetched by validity time; the row's base date is used
only to stamp the output, so the trajectory loader recovers
``(basetime, step)``.

.. literalinclude:: yaml/reduce-trajectories-analyses.yaml
   :language: yaml

Because analyses exist before the run starts, such a window may reach back
past the base date: a 24 h mean on a 6 h step covers ``base_date − 18h`` to
``base_date + 6h``, and the built variable records that as ``period: [-18h,
6h]``. That is legitimate, and it is why the restriction below applies only
to the run-anchored shape.

The run the layout imposes
==========================

``from: {base_dates: true, frequency: ...}`` — the samples are lead times of
the run initialised at the row's base date.  ``base_dates`` is a flag rather
than a table: the run is the layout's, so there is nothing to enumerate.

.. literalinclude:: yaml/reduce-trajectories-run.yaml
   :language: yaml

There is deliberately **no** ``steps:`` key. The lead times to fetch are
*derived* from the output steps and ``period``: the window of output step
``s`` needs ``s − period + k·frequency`` for ``k = 1…period/frequency``. With
output steps every 6 h and ``from.frequency: 1h``, a 6 h maximum at step 12
reads lead times 7…12 — denser than the output grid, and reaching below it.
Declaring ``steps:`` would state something the source works out, and stating
that they come from the layout would suggest the samples sit *on* the output
steps — the one thing that is not true. ``base_dates`` is a flag for the same
reason: with no ``steps:`` beside it, there is nothing for a sentinel value to
agree with.

Two rules follow from the run:

-  ``base_dates: true`` is only valid under ``layout: trajectories``. It
   inherits the run from the output layout, and no other layout imposes one.
-  ``steps.start`` must be at least ``period``. The window has to lie inside
   the run: reaching further back would need fields from before the forecast,
   which is analysis — a different quantity, not merely missing data. Use a
   base-less ``from:`` if that is what you want.

**********************
 Completeness
**********************

A window is reduced only if it is whole. Every sample of a sampled window, and
every field every subwindow of a covered one needs, is required. A reduction
over part of a window is not an incomplete answer but a wrong one — biased by
whatever is missing, and biasing the dataset statistics behind it — so a
window missing anything is an error that names the windows and what they still
wanted.

Interval-valued source data fails earlier where it can. A window the archive
cannot cover at all is refused while the recipe is resolved, before anything
is fetched, and the error says what the search tried: the intervals on offer
at the window's start and end, and the route that came closest. Only a window
that *could* be covered, and then did not receive one of its fields, reaches
the check above.

Windows reaching before ``dates.start`` request source data from before the
start of the dataset, exactly as ``accumulate`` does; that data simply has to
exist.

A field the source returns that belongs to no window is also an error. Under
``from: {frequency: ...}`` it usually means the cadence does not match what
the source provides; under an interval-valued ``from:`` it means the field
spans an interval the description did not declare.

This is stricter than ``accumulate``, deliberately. ``accumulate`` drops a
window that *no* field reached, with a warning, because MARS may answer an
interval request loosely (the ``scda``/``oper`` stream split); a window that
received only some of its fields is an error there too. These sources drop
nothing at all: a sum missing a part comes out visibly too small, while a mean
or a maximum over the fields that happened to arrive looks entirely plausible.

************
 Limitations
************

-  Reducing instantaneous fields from a *different* forecast archive — an
   explicit ``base_dates`` table with its own ``steps`` — is not implemented.
   It needs run selection ("which run serves this validity time"), and it is
   orthogonal to the output layout: it would apply to a gridded recipe just as
   much as to a trajectory one.
-  Differencing a from-start *mean* archive is refused. It is reconstructible
   in principle — ``m(6,9) = (9·m(0,9) − 6·m(0,6)) / 3`` — but not by the plain
   signed sum a subwindow performs; it needs length-weighted contributions and
   a renormalisation. No archive known to us stores one.
-  Summing over time is ``accumulate:``; there is no ``sum:`` block. The
   name ``sum`` already belongs to the anemoi-transform filter that sums
   *across variables*.
-  The window is always end-anchored. Centred windows would be a change to
   the dataset-wide anchoring convention, not to these sources.
