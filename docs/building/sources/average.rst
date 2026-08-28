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
     - **Required.** What the source data is, recognised by whether
       ``base_dates`` is present:

       -  ``{frequency: <cadence>}`` — **base-less**: instantaneous fields
          indexed by validity time, one every *cadence*. Works under either
          output layout.
       -  ``{base_dates: true, frequency: <cadence>}`` — the forecast run
          the trajectory layout imposes; see `Trajectories`_.

.. literalinclude:: yaml/reduce-average.yaml
   :language: yaml

``minimum`` and ``maximum`` take exactly the same keys — a daily maximum
of hourly 2 m temperature:

.. literalinclude:: yaml/reduce-maximum.yaml
   :language: yaml

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
it usually means ``from.frequency`` does not match what the source actually
provides.

************
 Limitations
************

-  Reducing instantaneous fields from a *different* forecast archive — an
   explicit ``base_dates`` table with its own ``steps`` — is not implemented.
   It needs run selection ("which run serves this validity time"), and it is
   orthogonal to the output layout: it would apply to a gridded recipe just as
   much as to a trajectory one.
-  Reducing *accumulated* source data is not implemented either; it would
   first have to be de-accumulated to increments.
-  Summing over time is ``accumulate:``; there is no ``sum:`` block. The
   name ``sum`` already belongs to the anemoi-transform filter that sums
   *across variables*.
-  The window is always end-anchored. Centred windows would be a change to
   the dataset-wide anchoring convention, not to these sources.
