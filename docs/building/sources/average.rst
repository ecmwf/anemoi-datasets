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

They are the instantaneous counterpart of :ref:`accumulate
<sources-accumulate>` and share its recipe shape key for key: ``source:``
says where the data comes from, ``period:`` says what you want, and
``from:`` says what the source data is.

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

This whole section is about instantaneous source data. For fields that each
span an interval, the window is covered rather than sampled and there is no
cadence to divide — see `Interval-valued source data`_.

***************
 Configuration
***************

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
     - **Required.** What the source data is. ``frequency:`` means a field
       *exists every* cadence; ``accumulation:`` or an explicit ``steps:``
       list means each field *spans* an interval. That distinction decides
       everything else:

       -  ``{frequency: <cadence>}`` — **base-less**: instantaneous fields
          indexed by validity time, one every *cadence*. Works under either
          output layout.
       -  ``{base_dates: true, frequency: <cadence>}`` — the forecast run
          the trajectory layout imposes; see `Trajectories`_.
       -  every interval-valued shape ``accumulate`` accepts — see
          `Interval-valued source data`_.
   * - ``over``
     - The length of each subwindow, for interval-valued source data only.
       Defaults to ``period``, i.e. the window is one piece. See
       `Interval-valued source data`_.

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

This is what makes an irregular archive expressible at all. ``rr/se-al-ec``
stores gusts over ``0-1, 1-2, 2-3, 3-6, 6-9`` — hourly to step 3,
three-hourly after — so there is no single ``frequency:`` to state, and the
step pairs are written out instead:

.. literalinclude:: yaml/reduce-maximum-gusts.yaml
   :language: yaml

The window is then covered by whole archived intervals rather than sampled,
and the reduction is applied across them.

Subwindows and ``over:``
========================

The pieces the window is cut into are **subwindows**, and ``over:`` is how
long each one is. It defaults to ``period`` — the window is one piece — which
is right whenever the archive already holds what you are asking for.

It matters when it does not. A precipitation archive stores a running total
from the start of the run, so a window is rebuilt by subtraction and a maximum
over one such piece is just that piece. ``over: 1h`` cuts the window into
hourly pieces and rebuilds each, which is the wettest hour in the window:

.. literalinclude:: yaml/reduce-maximum-over.yaml
   :language: yaml

The length is part of what you are asking for, not a hint: the largest hourly
total and the largest 3-hourly total are different numbers. That is why
``over:`` has to be stated rather than inferred from whatever the archive
happens to offer — and why stating it at the full ``period`` declares nothing
and changes nothing.

``accumulate`` does not take ``over:``. A sum of pieces is the same whatever
length they are.

What is refused
===============

Two checks, because a wrong reduction here is silent rather than loud.

**Before anything is fetched**, from the recipe, in two forms.

``maximum:`` over an archive whose window has to be rebuilt by subtraction is
refused. Subtracting two cumulative maxima does not give the maximum over the
difference of their intervals — if the extreme falls in the earlier part, both
fields carry it and the difference is zero. The error names the fields it would
have needed.

``maximum:`` over an archive that holds the whole window outright *as an
accumulation* is also refused, even though nothing is being subtracted. The
window is then one piece, and a maximum over one piece is that piece — so the
result would be the window's total with a ``max`` label on it. This is not a
corner case: ``od-oper`` runs 12 h apart with a 6 h ``period`` gives a window
held outright every other row, and a differenced one in between, so a recipe
without ``over:`` would be wrong in two different ways on alternating rows.

The same shape is accepted when the archive stores *maxima* — one field really
is the answer there. ``from:`` is what separates them: an ``accumulation``
scheme says the fields add up, while explicit step pairs describe an archive of
stored statistics.

``over:`` resolves both, because it declares the quantity is additive and says
at what length.

**When a field arrives**, from the field: one carrying a ``max``, ``min`` or
``avg`` is never rebuilt by subtraction, whatever the recipe said. This is the
check that catches ``accumulate`` over a gust archive, which declares no
statistic at all.

Neither can be dropped. The first reads a recipe, which can simply be wrong
about what the archive holds; the second reads the data, but only once a
retrieval has already happened.

Nothing recovers a finer interval than the archive stores. If the shortest
gust interval is ``[6,9]``, there is no ``[6,7]`` to be had — that information
is not in the data, and achievable windows are unions of whole archived
intervals.

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

Every sample of every window is required. A window missing a sample is an
error, never a mean over the samples that happened to be there: a short
mean is a silently biased field, and it would bias the dataset statistics
with it. Windows reaching before ``dates.start`` request source data from
before the start of the dataset, exactly as ``accumulate`` does; that data
simply has to exist.

A field returned by the source that belongs to no window is also an error —
it usually means ``from:`` does not match what the source actually provides.

This is stricter than ``accumulate``, deliberately. ``accumulate`` drops a
window no field reached, because MARS may answer an interval request loosely
(the ``scda``/``oper`` stream split). A missing sample under ``average:`` is a
mean over fewer fields than you asked for, which is a biased field and biased
dataset statistics behind it.

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
