.. _sources-mars:

######
 mars
######

The ``mars`` source will retrieve the data from the ECMWF MARS archive.
For that, you need to have an ECMWF account and build your dataset on
one of the Centre's computers, or use the ``ecmwfapi`` Python package.

The `yaml` block can contain any keys that follow the `MARS language
specification`_, with the exception of the ``date``, ``time``, and
``step``.

The missing keys will be filled with the default values, as defined in
the MARS language specification.

.. literalinclude:: yaml/mars1.yaml
   :language: yaml

Data from several level types must be requested in separate requests,
with the ``join`` command.

.. literalinclude:: yaml/mars2.yaml
   :language: yaml

See :ref:`naming-variables` for information on how to name the variables
when mixing single-level and multi-level variables in the same dataset.

Wildcard date / time filters
============================

For the (very common) case where a MARS request should be restricted to
a subset of base dates (e.g. "only 00Z runs", "only the 1st of the
month"), ``date`` and ``time`` can be given as *wildcard filters* rather
than as literal values:

.. code:: yaml

   mars:
     class: od
     expver: "0001"
     type: fc
     grid: 20./20.
     levtype: sfc
     stream: oper
     param: [10u]
     date: "????-??-01"       # only the 1st of the month
     time: 0                  # only the 00 UTC run
     step: "0/to/240/by/12"

Semantics:

-  When ``date`` is a string that contains ``?``, it is treated as a
   regex-like pattern against the *computed base date* in ``YYYYMMDD``
   form (``-`` is stripped, ``?`` is ``.`` in the regex). The MARS
   request sent over the wire never sees the wildcard.
-  Any accompanying ``time:`` value (or list of values) is interpreted
   as a set of ``HHMM`` base-time filters, again applied to the
   *computed* base time.
-  The filters act on the expansion of a validity-date request. They are
   rejected on the forecast and interval paths — a filter inside a
   :ref:`trajectory <layouts-trajectories>` recipe or inside
   ``accumulate:`` raises a clear error at build time.

The legacy ``user_date`` / ``user_time`` keys that some configurations
relied on are no longer accepted; a ``ValueError`` is raised pointing
at the wildcard shorthand above.

Trajectory recipes
==================

``mars:`` also handles the forecast-date and forecast-interval cases
required by the :ref:`trajectories <layouts-trajectories>` layout —
``date``, ``time`` and ``step`` are stamped from the
``(basetime, valid_time)`` pair (or the accumulation interval) chosen by
the pipeline. No special recipe key is required; the same MARS block is
used regardless of layout.

Hindcasts (reforecasts)
=======================

Reforecast streams (e.g. ``eefh``, ``enfh``) archive each field under
**two** dates: ``date`` is the *reference date* of the model run (a
recent date), and ``hdate`` is the start of the hindcast run — the same
month-day as the reference date in one of the previous years. The valid
time of a field is ``hdate + step``.

In a trajectory recipe the base dates are the hindcast starts (hdates).
The ``hindcast:`` option tells the ``mars`` source how to map each base
date back to its reference date at request time:

.. code:: yaml

   base_dates:
     start: 2003-06-29
     end: 2023-11-11
     frequency: 24h
     missing:
       # every day whose month-day matches no reference date
       - {start: 2003-06-30, end: 2003-07-02}
       # ...

   steps:
     start: 12
     end: 360
     frequency: 12

   input:
     mars:
       class: od
       expver: "0001"
       stream: eefh
       type: cf
       levtype: sfc
       grid: o96
       param: [2t]
       hindcast:
         reference_start: 2023-06-29   # reference-date window
         reference_end: 2024-11-11
         day_of_week: [monday, thursday]
         years: 20                     # hindcast years per reference date

   output:
     layout: trajectories

For each base date the source finds the unique reference date whose
hindcast calendar (same month-day, up to ``years`` years back) contains
it, and sends ``date=<refdate>, hdate=<basetime>, time=0000`` with the
step unchanged (relative to the hdate). The mapping must be
unambiguous: if a base date is a hindcast date of two reference dates
(e.g. a window longer than a year without a ``day_of_week``
restriction), the recipe is rejected.

The option works both for instantaneous fields and inside
``accumulate:`` blocks (hindcast tp/cp are from-zero accumulations).
Since the ``hdate`` key is part of the mars metadata of the retrieved
fields, add it to the accumulate grouping exclusions:

.. code:: yaml

   accumulate:
     accumulated: from-zero
     period: 12h
     group_by:
       namespace: mars
       ignore: [date, time, step, hdate]
     source:
       mars:
         # ... as above, with the same hindcast block
         param: [tp, cp]

.. _mars language specification: https://confluence.ecmwf.int/display/UDOC/MARS+user+documentation
