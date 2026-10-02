.. _sources-reduce:

##########
 reduce
##########

The ``reduce`` source builds one output field per validity date by applying a
single operation — ``sum``, ``max``, ``min`` or ``mean`` — to the archived
fields that cover ``[valid_date - period, valid_date]``.

It is the source to use for quantities that can be aggregated **directly**. The
archetypal case is a wind gust field, which holds the largest gust observed
since the last output: a 6-hourly dataset built from hourly gusts wants the
maximum of the six hourly fields, not their sum.

.. code:: yaml

   input:
     reduce:
       period: 6h
       operation: max          # sum (default) | max | min | mean
       covering:
         auto:
           - [0,  "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12"]
           - [12, "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12"]
       source:
         mars:
           class: od
           stream: oper
           type: fc
           levtype: sfc
           param: [10fg]

For a valid date of 12:00 this takes the maximum of the six hourly fields
covering 06:00 → 12:00.

Parameters
==========

- **period**: the window to reduce over (e.g. ``6h``, ``12h``, ``1d``).
  Periods shorter than one hour are not supported.
- **operation** (optional, default ``sum``): ``sum``, ``max``, ``min`` or
  ``mean`` (also spelled ``avg`` / ``average``). See `Operations`_.
- **source**: the data source configuration. One of ``mars``, ``fdb`` or
  ``grib-index``.
- **covering**: how the archive lays out its intervals. Same shapes as the
  ``accumulate`` source — see
  :ref:`Specifying covering of accumulation intervals <sources-accumulate>`.
- **patch** (optional): metadata patches applied to the fields returned by the
  source; see
  :ref:`Fixing mis-encoded step metadata <sources-accumulate>`.
- **group_by** (optional): which metadata identifies a field group, as for
  ``accumulate``.

.. _sources-reduce-vs-accumulate:

reduce or accumulate?
=====================

Both build a value for a window out of archived fields, but they are allowed to
do different things to get there.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * -
     - ``accumulate``
     - ``reduce``
   * - Combines with
     - addition only
     - ``sum``, ``max``, ``min``, ``mean``
   * - May subtract archived fields
     - **yes** — ``a(6,12) = +a(0,12) − a(0,6)``
     - no, never
   * - Needs the window tiled exactly
     - no
     - **yes**
   * - Works on accumulated-from-start archives
     - yes
     - no — see below
   * - Trajectory recipes
     - yes
     - not yet

Use ``accumulate`` for additive quantities such as precipitation, especially
where the archive stores values accumulated from the start of the forecast: only
``accumulate`` can reconstruct a window by subtraction. Use ``reduce`` when the
value for the window is obtained by applying one operation directly to fields
that already tile it.

Operations
==========

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Value
     - Meaning
   * - ``sum``
     - Default. Adds the covering fields.
   * - ``max``
     - Largest value over the window, element by element (e.g. maximum wind gust).
   * - ``min``
     - Smallest value over the window.
   * - ``mean``
     - Time-weighted average over the window. Also spelled ``avg`` or ``average``.

``mean`` weights each contributing field by the length of the interval it
covers::

   mean = sum(length_i * values_i) / sum(length_i)

This matters whenever the covering mixes interval lengths. An archive that is
hourly out to step 6 and 3-hourly beyond it can tile a window with a 1h piece
and a 3h piece; a plain unweighted average of the two would be wrong. Because
``reduce`` requires the intervals to tile the window exactly, the weights sum to
the requested period and nothing is double counted. With a uniform covering the
result is the ordinary arithmetic mean.

Instantaneous fields are not supported
======================================

``reduce`` combines fields that each cover a *slice* of the window. An
instantaneous field — a snapshot such as ``2t`` or ``10u`` from an analysis —
covers no interval at all (``startStep == endStep``) and so tiles nothing.

This is the most likely way to misuse the source, because "6-hourly mean 2t"
sounds like something it should do. It is a genuinely different operation:
averaging *snapshots* sampled at a few times is not the same as averaging a
statistic that is defined over the whole window, and ``reduce`` does not do it.

Rather than guess, the source rejects such a field the first time it sees one,
naming the two things that could be wrong:

.. code:: text

   Field GribField(2t,20210101,0000,6,instant) covers no time interval:
   startStep == endStep == 6 (stepType='instant'). This source combines fields
   that each cover a slice of the window, so every field must carry a real
   interval.
   - If this parameter is instantaneous (a snapshot, e.g. 2t or 10u), it cannot
     be aggregated this way. ...
   - If it really is an interval statistic whose metadata is wrong, repair
     startStep with a 'patch:' entry ...

The second case is real: some archives encode an interval statistic with
``startStep == endStep``. ``accumulate`` reads such a field as covering
``[0, endStep]``, which is right for an accumulation-from-start archive but
would silently turn a snapshot into a window statistic. ``reduce`` therefore
refuses the shortcut and asks you to state the interval explicitly with a
patch. See :ref:`Fixing mis-encoded step metadata <sources-accumulate>`.

What ``reduce`` requires of the archive
=======================================

``reduce`` never subtracts, so the covering it resolves must be a **gapless,
forward-only tiling** of the window: every interval positive, their union
exactly ``[valid_date - period, valid_date]``, nothing hanging outside it.

That restriction is what makes ``max`` and ``min`` definable at all. Addition is
invertible — a window can be recovered by subtracting archived fields — but
extrema are not: the maximum over ``[6, 12]`` cannot be recovered from the
maxima over ``[0, 12]`` and ``[0, 6]``. The same goes for ``mean``: a window average cannot be
recovered from two overlapping averages without knowing what to subtract.
``reduce`` applies the rule to ``sum`` as well, for consistency of scope:
reconstructing windows by differencing is ``accumulate``'s job.

In practice:

- An archive storing per-step values (ERA5-style,
  ``accumulated-from-previous-step``, or an explicit ``0-1/1-2/...`` interval
  list) works.
- An archive storing values from the start of the forecast does **not**. Getting
  a window out of it would mean de-aggregating first, which is out of scope. The
  source detects this while resolving the covering — before any data is
  retrieved — and fails with an explanatory error.

.. warning::

   The ``covering: {auto: ...}`` presets describe how **precipitation** is laid
   out for a given class/stream. A parameter such as wind gust may be archived
   with entirely different step ranges in the very same stream, so give an
   explicit interval list rather than relying on ``auto``.

NaNs propagate through every operation, as they do for ``sum``.

Extrema and GRIB edition 1
==========================

This section is about *encoding* the result, and only concerns fields that got
that far. Two questions are asked in order:

1. Does each contributing field cover a time interval? If not, the field is
   rejected whatever the operation — see
   `Instantaneous fields are not supported`_. An instantaneous ``2t`` never
   reaches the encoding step at all.
2. Can the output's GRIB edition express the operation? If not, the output is
   promoted to GRIB2, as described below.

``sum`` and ``mean`` are safe in both GRIB editions. ``max`` and ``min`` are
restricted in edition 1, and the restriction depends on the **parameter**.

GRIB1 cannot state "maximum" or "minimum" in the product definition. Its time
range indicator (table 5) has codes for average (3) and accumulation (4) but
none for extrema, so eccodes writes both as ``timeRangeIndicator=2`` — "valid
over the interval", statistic unspecified — and reads that back as a maximum
whatever was written.

In GRIB1 the statistic is carried by the parameter instead: ``10fg`` *is* a
maximum, ``mn2t6`` *is* a minimum, ``tp`` *is* an accumulation. eccodes exposes
this as the parameter-derived key ``stepTypeForConversion``, and that is what
anemoi falls back on when the message itself says nothing. So on edition 1 an
extremum is only expressible when the parameter declares that same statistic.

When it is not, ``reduce`` **writes the output as GRIB2** rather than emitting a
field whose recorded statistic would be wrong or missing:

.. list-table::
   :widths: 22 14 64
   :header-rows: 1

   * - Parameter declares
     - Operation
     - Output
   * - ``max`` (e.g. ``10fg``)
     - ``max``
     - Edition 1, unchanged.
   * - ``min`` (e.g. ``mn2t6``)
     - ``min``
     - Edition 1, unchanged.
   * - ``max`` (e.g. ``10fg``)
     - ``min``
     - Promoted to edition 2. In edition 1 it would have been recorded as a
       *maximum*.
   * - something else, or nothing (e.g. ``tp``, which declares ``accum``)
     - ``max`` or ``min``
     - Promoted to edition 2. In edition 1 anemoi could not have classified the
       field at all, failing later in the build.

The GRIB written by this source is an internal intermediate — the dataset itself
is unaffected by the edition — and the MARS keys (``class``, ``stream``,
``expver``, ``type``) survive the conversion. Promotion is logged once per block.

.. warning::

   Promotion **renames the output parameter**. eccodes resolves the promoted
   message to the parameter that genuinely matches the statistic, so a 6h
   minimum of ``10fg`` comes out as ``min_i10fg`` ("Time-minimum 10 metre wind
   gust"). A ``rename:`` in the pipe keyed on the original name will not match.

   Often the promoted name is the right one: the 6h maximum of an hourly-mean
   ``2t`` comes out as ``mx2t``. But for combinations with no canonical GRIB2
   parameter — the maximum of an accumulated quantity such as ``tp``, say — it
   is ``unknown``. The statistic and the values are still correct, but you will
   want a ``rename:`` to give the variable a sensible name.

.. note::

   One gap inherited from ``accumulate``: ``sum`` on a GRIB1 parameter that does
   not itself declare ``accum`` (summing ``10fg``, for instance) writes the
   historical encoding, from which anemoi records no ``process`` or ``period``
   at all. The values are correct; only that provenance is missing. ``sum`` on a
   genuine accumulation parameter, which is the ordinary case, is unaffected.
