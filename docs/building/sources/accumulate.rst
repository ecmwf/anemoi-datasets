.. _sources-accumulate:

############
 accumulate
############

**********
 Concepts
**********

Dataset to build versus source data
====================================

``accumulate`` reads **source data** and writes into the **anemoi dataset** being
built. The job of the source is to translate one into the other.

.. mermaid::

   flowchart LR
       s[source data] -->|accumulate| d[anemoi dataset]

Both hold accumulations, each stamped with a timestamp and accumulated over some period. The anemoi dataset
follows some fixed conventions (accumulated over a
fixed interval, and end-anchored see below) — while the source data may use a different scheme.

.. note::

   ``accumulate`` sums over time. To average, minimise or maximise a window
   of *instantaneous* fields instead, see :ref:`average, minimum and maximum
   <sources-average>`, which share this source's ``source:`` / ``period:`` /
   ``from:`` shape.


Accumulation interval
---------------------

In an anemoi dataset, the accumulation interval is always a fixed accumulation period, (for a given variable) :
the requested ``period:``.

Source data may use a different accumulation period, or even a different scheme.
The source data may be accumulated :

 - a fixed period (e.g. 1h, 3h, 6h, 12h, 24h)
 - accumulated from the start of the forecast
 - accumulated from the previous step
 - accumulated from earlier but reset every *N* hours (special custom use case)
 - other schemes (managed by a lookup table)

Source data accumulation interval is described by the ``accumulation:`` key in the ``from:`` block.

The ``accumulate`` source combines them to produce the fixed period you request.


.. _the-anchor-convention:

Anchoring
---------

An accumulation over a time interval ``[start, end]`` is associated with a
timestamp — the "date" of the accumulation. Its **anchor** says which point
of the interval that timestamp names

This anchor is used in the anemoi dataset as :

- the ``valid_datetime`` in the :ref:`gridded layout <layouts-gridded>`
- the ``base_date + step`` in the :ref:`trajectories layout <layouts-trajectories>`.

There a three well know time of anchoring : **end-anchored** the end of the interval, **start-anchored** the start, **mid-anchored** the middle ``(start + end) / 2``.

//todo do a nice diagram here

In an anemoi dataset, the convention is that the data is always **end-anchored**.

Source data must also be **end-anchored**. Source data using a different anchoring scheme are not supported yet,
but is likely to be needed in the future.

There is no options in ``accumulate:`` to change the anchoring as the end-anchoring is the only one supported.


Wrong metadata vs different convention
--------------------------------------

A source that uses a different anchor or scheme is **not wrong** — it is a
self-consistent convention, to be *declared* (through ``from:`` and potentially with
``patch:``) so ``accumulate`` can translate it, additional code may be needed to do so.
That is distinct from source data whose ``startStep``/``endStep`` metadata is genuinely **wrong**, which
``patch:`` *fixes*. See :ref:`how-to-patch`.

***************
 Configuration
***************

Three keys configure the source. ``source:`` says where the data comes
from; then ``period:`` is what you want, and ``from:`` is what the source
data is.

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - key
     - meaning
   * - ``period``
     - **Required.** The accumulation period you want (``6h``, ``12h``,
       ``1d``, …). Periods shorter than one hour are not supported yet.
   * - ``source``
     - **Required.** The data source, as a single-key dictionary
       (``source: {mars: {...}}``). ``mars`` and ``grib-index`` are
       supported.
   * - ``from``
     - *Optional.* What the source data is.

       -  **omitted** — recognised from the ``mars`` source; this is the
          common case, see :ref:`well-known-archives`.
       -  :ref:`base_dates + steps <trajectories>` — indexed by base date ×
          step (MARS-like). ``steps`` is a regular range plus an
          ``accumulation`` (``from-zero`` / a duration / reset), or an
          explicit list of ``"sA-sE"`` pairs for an irregular grid.
       -  :ref:`lookup-table <how-to-lookup-table>` — an explicit table, the
          escape hatch; the table *is* the description.
       -  :ref:`accumulation alone <valid-time>` — the *bare* form: indexed
          by validity time alone (``grib-index``), or the run imposed by a
          trajectory layout. ``accumulation`` is a duration — the length
          each field holds.
   * - ``patch``
     - *Optional.* For source data with a different anchor, or whose step
       metadata is wrong; see :ref:`how-to-patch`.
   * - ``group_by``
     - *Optional.* Which fields are accumulated together; see
       :ref:`how-to-group-by`.

The **bare** form means different things in different places: outside a
``layout: trajectories`` recipe it describes base-less source data indexed
by validity time, so ``accumulation`` must be a duration; inside one it
describes the run the layout imposes, and all three values are allowed
(see :ref:`accumulate-trajectories`).

.. note::

   The older spellings — the ``availability:`` and ``covering:`` keys,
   and the block-level ``accumulation:`` flag of trajectory recipes —
   are accepted for one release with a ``DeprecationWarning``. Run
   ``anemoi-datasets recipe --migrate <recipe>`` to rewrite old recipes
   automatically (this also converts the even older ``accumulations``
   source).

**************
 Common cases
**************

.. _well-known-archives:

A well-known MARS archive
=========================

Write **no** ``from:`` at all — this is the common case. The layout of
the source data is recognised from the ``mars`` source parameters
(class, stream, origin):

.. literalinclude:: yaml/accumulate-auto-era5.yaml
   :language: yaml

The combinations recognised are:

-  ERA5 reanalysis (class ``ea``, stream ``oper``)
-  ERA5 ensemble data assimilation (class ``ea``, stream ``enda``)
-  ECMWF operational forecasts (class ``od``, stream ``oper``)
-  ECMWF operational ensemble data assimilation (class ``od``, stream
   ``elda``)
-  CERRA regional reanalysis (class ``rr``, origins ``se-al-ec`` and
   ``fr-ms-ec``)
-  ERA5-Land (class ``l5``, stream ``oper``)

Only a ``mars`` source can be recognised this way, and only for the list
above; any other source data must be described with one of the forms
below, and the error says so.

.. _trajectories:

Forecast source data: ``base_dates`` + ``steps``
================================================

Most accumulated source data comes in **forecast trajectories**: model
runs initialised at recurring base dates, archiving fields on a grid of
forecast steps, with a native accumulation scheme.  Writing
``base_dates:`` and ``steps:`` in ``from:`` describes exactly these three
facts, reusing the ``base_dates:`` / ``steps:`` vocabulary of the
:ref:`trajectories layout <layouts-trajectories>`.

ECMWF operational forecasts, accumulated from the start of the forecast:

.. literalinclude:: yaml/accumulate-trajectories-od-oper.yaml
   :language: yaml

ERA5, whose fields are per-step increments:

.. literalinclude:: yaml/accumulate-trajectories-era5.yaml
   :language: yaml

Given the description, the package searches for the combination of
archived intervals covering each requested window — including summing
increments, subtracting from-zero accumulations (e.g. ``a(0,12) −
a(0,6)``), and switching between model runs when a window spans more than
one trajectory.

The three keys
--------------

-  ``base_dates``: the recurring initialisation times.

   -  ``times`` (required): list of initialisation times (``["06:00",
      "18:00"]``; bare hours like ``[6, 18]`` are accepted).
   -  ``day_of_month`` (optional): restrict to given days of the month
      (e.g. ``1`` for monthly runs).
   -  ``day_of_week`` (optional): restrict to given days of the week
      (e.g. ``[mon, thu]``).
   -  ``start`` / ``end`` (optional): bound truly bounded source data
      (e.g. the experiment only ran over 2016–2020).

   A wildcard string in the :ref:`from-trajectories
   <sources-from-trajectories>` *source* dialect is accepted as sugar and
   converted to the structured form, e.g. ``base_dates: "????-??-01
   00:00"`` is ``{times: ["00:00"], day_of_month: 1}``.

-  ``steps``: **the fields the source holds**, written one of two ways.

   **A regular range** ``{start, end, frequency}`` (the trajectories-layout
   shape) *plus* an ``accumulation`` scheme (below). ``frequency`` is the step
   *spacing* (where fields are); with a **duration** it is independent of the
   accumulation *length* (how long each field is). The common case is that
   they are equal — contiguous per-step increments, the field at step *s*
   covering ``(s − duration, s)``. ERA5 is hourly out to 18 h:

   .. code:: yaml

      steps: {start: 1h, end: 18h, frequency: 1h}   # fields at steps 1..18
      accumulation: 1h                              # each field spans 1h

   They need not be equal: ``accumulation`` larger than ``frequency`` is an
   overlapping (rolling) archive — a 24 h window every 6 h is
   ``{start: 24h, end: 240h, frequency: 6h}`` + ``accumulation: 24h``; smaller
   is a sparse grid with gaps. The only constraint is ``start >= accumulation``
   (the first field cannot begin before the forecast). There is no field at
   step 0.

   **An explicit list of pairs**, one per available field (whole hours), each
   written ``"sA-sE"`` or ``[sA, sE]`` (the two are interchangeable). The
   pairs *are* the description, so ``accumulation`` is not used (and is
   rejected). This is the only form general enough for an **irregular grid of
   mixed accumulation lengths** — CERRA, for instance, stores from-zero fields
   hourly to 6 h then 3-hourly to 30 h:

   .. code:: yaml

      steps: ["0-1", "0-2", "0-3", "0-4", "0-5", "0-6",
              "0-9", "0-12", "0-15", "0-18", "0-21", "0-24", "0-27", "0-30"]

   and an archive whose increments coarsen (1 h pieces to 6 h, 3 h after) —
   which no single ``accumulation`` can express — is simply its pairs:

   .. code:: yaml

      steps: ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-9", "9-12"]

-  ``accumulation``: how the source data accumulates — required with a
   ``steps`` **range**, omitted (and rejected) with an explicit **pair
   list**. One of:

   -  ``from-zero`` — fields are accumulated from the start of the
      forecast (``a(0, step)``), e.g. ECMWF operational forecasts;
   -  a **duration** (e.g. ``1h``) — each field is a window that long,
      ``a(step − 1h, step)``, e.g. ERA5. It is the window *length*,
      independent of ``steps.frequency`` (the spacing); ``start`` must be
      ≥ the duration;
   -  ``from-zero-reset-every-<frequency>`` — from-zero accumulations
      that restart every ``<frequency>`` of *lead time*; see
      :ref:`how-to-reset`.

Given the description, ``accumulate`` knows exactly which fields the
source holds and combines them to produce each requested window.

.. _valid-time:

Base-less source data: a bare ``accumulation``
==============================================

Some source data is **base-less**: fields are indexed by validity time only,
each field being the accumulation over the fixed-length window ending at
its validity time. This is the shape of a ``grib-index`` store of
pre-computed increments. Write ``accumulation:`` alone — no ``base_dates``
or ``steps`` — and it is that window length; it must divide the requested
``period`` (the windows are summed) and divide 24 h. This form is rejected
with a ``mars`` source — MARS fields are anchored to a model run, so they
are never indexed by validity time alone (add ``base_dates`` and
``steps``).

.. literalinclude:: yaml/accumulate-valid-time.yaml
   :language: yaml

When the source data's ``accumulation`` equals the ``period`` there is
nothing to sum: ``accumulate`` still validates the
``startStep``/``endStep`` metadata against the request and re-stamps the
output uniformly.

A base-less source can fill a ``layout: trajectories`` dataset too. ``from:``
describes the *source data*; the output layout decides the *output* — the two
are orthogonal, and ``accumulate`` bridges them. Given the same bare
``accumulation:`` duration, the field valid at ``base_date + step`` is
reused for that ``(base_date, step)`` row (overlapping rows share the same
physical field, since base-less data carries no run of its own).

.. _accumulate-trajectories:

Forecast accumulations (trajectory recipes)
===========================================

Inside a :ref:`trajectory recipe <layouts-trajectories>`,
``accumulate:`` produces per-step accumulation fields anchored on the
basetime imposed by the layout.

``from:`` describes the *source data*; the trajectory layout decides the
*output* — the two are orthogonal, so every ``from:`` shape is accepted here,
each bridged onto the imposed ``(base_date, step)`` rows:

-  **the layout's own run** — ``base_dates: from-layout`` and
   ``steps: from-layout`` (the sentinel value, on **both** keys together): the
   source data *is* the run the layout imposes, so its grid is inherited and
   only the ``accumulation:`` scheme is stated. All three schemes are
   supported:

   -  ``from-zero`` — the source data stores accumulations from the basetime
      (``a(0, step)``). The window ``[bt+sA, bt+sE]`` is reconstructed as
      ``+a(0, sE) − a(0, sA)``.
   -  a **duration** (e.g. ``3h``) — the source data stores per-step
      increments that long. The requested ``period`` is re-accumulated by
      summing them, so ``period`` must be a whole multiple of the duration
      (``period: 6h`` from ``3h`` increments sums two fields; equal means no
      re-accumulation).
   -  ``from-zero-reset-every-<frequency>`` — from-zero accumulations
      restarting every ``<frequency>`` of lead time.

-  **a base-less, valid-time source** — a bare ``accumulation:`` *duration*
   (no ``base_dates``/``steps``): the field valid at ``base_date + step`` is
   summed to ``period`` and relabelled onto that row (see `Base-less source
   data`_).
-  **a different forecast archive** — explicit ``base_dates`` + ``steps`` (or
   an omitted ``from:`` for a well-known MARS archive, or a ``lookup-table``)
   describing an archive whose run grid differs from the layout's: each output
   window is reconstructed from that archive's runs and stamped at the layout's
   ``(base_date, step)``.

Recipe validation additionally requires ``steps.start >= period``: output
windows must not straddle the basetime.

.. code:: yaml

   base_dates: {start: 2021-01-01, end: 2021-01-03, frequency: 12h}
   steps:      {start: 6h, end: 30h, frequency: 3h}

   input:
     join:
       - mars: {type: fc, class: od, param: [q, t], levtype: pl,
                level: [50], stream: oper, grid: 20./20., expver: "0001"}
       - pipe:
           - accumulate:
               period: 1h
               from:
                 base_dates: from-layout
                 steps: from-layout
                 accumulation: from-zero
               source:
                 mars:
                   type: fc
                   expver: "0001"
                   class: od
                   grid: 20./20.
                   param: [tp]
                   levtype: sfc
                   stream: oper
           - rename:
               param: {tp: tp_accum_1h}

   output:
     layout: trajectories

.. note::

   For ECMWF forecasts, the forecasts at 00Z and 12Z are from the stream
   ``oper`` while the forecasts at 06Z and 18Z are from the stream
   ``scda``.

********
 How to
********

The cases below are awkward source data, not everyday configuration. Reach
for them only when one of the common cases above cannot describe your
data.

.. _how-to-reset:

Archives whose accumulations reset
==================================

Some source data stores from-zero accumulations that restart at fixed
intervals of lead time. This is part of the (correct) data layout and is
described by the ``accumulation:`` value; it is *not* the same thing as
broken metadata (see :ref:`how-to-patch` — the example below needs both):

.. literalinclude:: yaml/accumulate-reset.yaml
   :language: yaml

.. _how-to-patch:

Source data with a different anchor, or wrong ``startStep``/``endStep``
=======================================================================

Keep two different things apart here — only one of them is a defect.

**A different anchor.** As explained in :ref:`the-anchor-convention`, the
dataset is end-anchored, but the source data need not be. A source that is
start-anchored or mid-anchored is *correct*; it simply names its
accumulations by a point of the interval other than the end. Reading it is
about declaring that anchor so ``accumulate`` can re-anchor each field onto
the end-anchored dataset — not about fixing anything.

   Today this is a direction, not a knob: ``accumulate`` assumes the source
   data is also end-anchored (the reconstruction below checks each field's
   valid date against the *end* of its reconstructed interval), and no
   shipped patch shifts the anchor. Start- and mid-anchored source data are
   the general case the same machinery is meant to grow into.

**Wrong ``startStep``/``endStep``.** Separately, some source data simply
encodes its step metadata incorrectly. The reconstruction relies on the
``startStep``/``endStep`` of the fields returned by the source; the
``patch:`` key applies named fixes to each field before it is matched to the
requested intervals:

.. note::

   The ``startStep``/``endStep`` naming here follows the MARS/GRIB
   vocabulary. Further development will abstract this so that it becomes
   agnostic to the GRIB/NetCDF difference.

-  ``set_start_step_to_zero`` — the source data encodes
   ``startStep == endStep`` but fields are accumulated from the start of
   the forecast;
-  ``reset_24h_accumulations`` — same, for source data whose accumulations
   reset every 24 h of lead time (see :ref:`how-to-reset`).

``patch:`` describes what the fields' metadata *wrongly says*;
``from.accumulation:`` describes what the data *actually is*. They are
orthogonal and one source may need both. Additional patches may be needed
for special use cases.

.. warning::

   If the data provided by the source does not match its
   description, the package checks the metadata of the source dataset and
   fails if the accumulation periods cannot be reconstructed. **If the
   metadata is incomplete or inconsistent, the package may produce
   incorrect results.**

.. _how-to-group-by:

Accumulating fields whose metadata differs
==========================================

It is possible to control the fields accumulated together through their
metadata. The ``group_by`` keyword allows to ignore some metadata when
deciding to group fields to accumulate them together. Ignored keys mean
that fields with different values will be accumulated together. Note that
``date``, ``time`` and ``step`` must always be ignored.

.. literalinclude:: yaml/accumulate-groupby.yaml
   :language: yaml

.. _how-to-lookup-table:

Step layouts that do not factorise
==================================

The escape hatch for source data whose layout cannot be written as
``base_dates × steps``: put the table under a ``lookup-table:`` key inside
``from:``. Each requested window is mapped to a fixed
``(base time, steps)`` entry, keyed by the window's offset (in hours)
inside a repeating cycle anchored at ``start:``. The requested ``period``
must exactly match a table key.

.. literalinclude:: yaml/accumulate-lookup-table.yaml
   :language: yaml

The entry pins **which** archived intervals may be used — that is the
point of the escape hatch, there is no search over base times. **How**
they combine is left to the same signed search as every other form, so an
entry may name several intervals and they are combined, not blindly
summed. An entry naming two from-zero fields therefore expresses a
difference::

   "6-12": [0, "0-12/0-6"]     # window 06->12 as +a(0,12) - a(0,6)

A set of intervals that cannot be combined into exactly the requested
window is an error, never a silent partial sum.
