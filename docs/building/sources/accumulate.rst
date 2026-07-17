.. _sources-accumulate:

###############
 accumulate
###############

Accumulations and flux variables, such as precipitation, are often
forecast fields, which are archived for a given base date (or reference
time) and a forecast time (or step). These fields are valid at the
forecast time and are accumulated over a given period of time, with the
relation: :math:`valid\_date = base\_date + step`.

Because the package builds datasets according to the valid date of the
fields, it must be able to reconstruct the requested accumulation period
from the available data in the source dataset. Furthermore, some fields
are accumulated since the beginning of the forecast (e.g. ECMWF
operational forecast), while others are accumulated since the last time
step (e.g. ERA5).

The ``accumulate`` source takes:

-  **period**: The requested accumulation period (e.g., ``6h``, ``12h``,
   ``24h``, ``1d``). This can be specified as a string with units
   (``"6h"``). Periods shorter than one hour such as ``"30min"`` are not
   supported yet.

-  **source**: The data source configuration, as a single-key nested
   dictionary (e.g. ``source: {mars: {...}}``). Currently ``mars`` and
   ``grib-index`` sources are supported.

-  exactly **one description key** telling the package how
   accumulations are stored in the archive:

   -  `from-trajectories`_ — the archive contains forecast runs
      (MARS-like);
   -  `from-increments`_ — the archive is base-less and valid-time
      indexed (``grib-index``);
   -  `from-lookup-table`_ — an explicit escape hatch for step layouts
      too irregular for the other two.

-  **patch** (optional): Patches to apply to fields returned by the
   source to fix metadata issues; see `Patching wrong metadata`_.

-  **group_by** (optional): Controls which fields are accumulated
   together; see `Controlling the fields regrouped within accumulation`_.

In a :ref:`trajectory recipe <layouts-trajectories>` there is **no**
description key — the layout imposes the base time — and the
``accumulated:`` scheme is declared directly on the accumulate block;
see `Forecast accumulations (trajectory recipes)`_.

.. note::

   The pre-redesign spellings — the ``availability:`` and ``covering:``
   keys, and the ``accumulation:`` flag of trajectory recipes — are
   accepted for one release with a ``DeprecationWarning``. Run
   ``anemoi-datasets recipe --migrate <recipe>`` to rewrite old recipes
   automatically (this also converts the even older ``accumulations``
   source).

.. _from-trajectories:

*******************
 from-trajectories
*******************

Most accumulation archives are made of **forecast trajectories**: model
runs initialised at recurring base dates, archiving fields on a grid of
forecast steps, with a native accumulation scheme. The
``from-trajectories:`` key describes exactly these three facts, reusing
the ``base_dates:`` / ``steps:`` vocabulary of the
:ref:`trajectories layout <layouts-trajectories>`:

-  ``base_dates``: the recurring initialisation times.

   -  ``times`` (required): list of initialisation times (``["06:00",
      "18:00"]``; bare hours like ``[6, 18]`` are accepted).
   -  ``day_of_month`` (optional): restrict to given days of the month
      (e.g. ``1`` for monthly runs).
   -  ``day_of_week`` (optional): restrict to given days of the week
      (e.g. ``[mon, thu]``).
   -  ``start`` / ``end`` (optional): bound a truly bounded archive
      (e.g. the experiment only ran over 2016–2020).

   A wildcard string in the ``from-trajectories`` *source* dialect is
   accepted as sugar and converted to the structured form, e.g.
   ``base_dates: "????-??-01 00:00"`` is ``{times: ["00:00"],
   day_of_month: 1}``.

-  ``steps``: the forecast step grid, in the same
   ``{start, end, frequency}`` shape as the trajectories layout. A
   *list* of such ranges is accepted for irregular grids, e.g.::

      steps:
        - {start: 1h, end: 6h,  frequency: 1h}
        - {start: 6h, end: 30h, frequency: 3h}

-  ``accumulated``: the archive's native accumulation scheme. One of:

   -  ``from-zero`` — fields are accumulated from the start of the
      forecast (``a(0, step)``), e.g. ECMWF operational forecasts;
   -  ``from-previous-step`` — fields are per-step increments
      (``a(previous step, step)``), e.g. ERA5;
   -  ``from-zero-reset-every-<frequency>`` — from-zero accumulations
      that restart every ``<frequency>`` of *lead time* (e.g.
      ``from-zero-reset-every-24h`` for research runs that reset daily).

Given the description, the package searches for the combination of
archived intervals covering each requested window — including summing
increments, subtracting from-zero accumulations (e.g. ``a(0,12) −
a(0,6)``), and switching between model runs when a window spans more
than one trajectory.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ECMWF operational (accumulated from zero)
     - ERA5 (accumulated from previous step)
   * - .. literalinclude:: yaml/accumulate-from-trajectories-od-oper.yaml
          :language: yaml
     - .. literalinclude:: yaml/accumulate-from-trajectories-era5.yaml
          :language: yaml

Automatic description for well-known archives
=============================================

For well-known MARS archives, ``from-trajectories: auto`` infers the
description from the ``mars`` source parameters (class, stream, origin).
Supported combinations are:

-  ERA5 reanalysis (class ``ea``, stream ``oper``)
-  ERA5 ensemble data assimilation (class ``ea``, stream ``enda``)
-  ECMWF operational forecasts (class ``od``, stream ``oper``)
-  ECMWF operational ensemble data assimilation (class ``od``, stream
   ``elda``)
-  Regional reanalysis (class ``rr``, origins ``se-al-ec`` and
   ``fr-ms-ec``)
-  ERA5-Land (class ``l5``, stream ``oper``)

Automatic detection is only supported for the ``mars`` source.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - ECMWF operational
     - ERA5
   * - .. literalinclude:: yaml/accumulate-auto-od-oper.yaml
          :language: yaml
     - .. literalinclude:: yaml/accumulate-auto-era5.yaml
          :language: yaml

Archives with a reset frequency
===============================

Some archives store from-zero accumulations that restart at fixed
intervals of lead time. This is part of the (correct) data layout and is
described by the ``accumulated:`` value; it is *not* the same thing as
broken metadata (see `Patching wrong metadata`_ — the archive below
needs both):

.. literalinclude:: yaml/accumulate-reset.yaml
   :language: yaml

.. _from-increments:

*****************
 from-increments
*****************

Some archives are **base-less**: fields are indexed by validity time
only, each field being the accumulation over the fixed-length window
ending at its validity time. This is the shape of a ``grib-index`` store
of pre-computed increments. The value is the increment length; it must
divide the requested ``period`` (the increments are summed), and it is
rejected with a ``mars`` source (MARS fields are anchored to a model run
— use ``from-trajectories:``).

.. literalinclude:: yaml/accumulate-from-increments.yaml
   :language: yaml

When the increment equals the ``period`` there is nothing to sum:
``accumulate`` still validates the ``startStep``/``endStep`` metadata
against the request and re-stamps the output uniformly.

.. _from-lookup-table:

*******************
 from-lookup-table
*******************

The escape hatch for step layouts that do not factorise into
``base_dates × steps``. Each requested window is mapped to a fixed
``(base time, steps)`` entry, keyed by the window's offset (in hours)
inside a repeating cycle anchored at ``start:``. No search is performed:
the requested ``period`` must exactly match a table key, and entries are
positive intervals only.

.. literalinclude:: yaml/accumulate-from-lookup-table.yaml
   :language: yaml

**********************************************
 Patching wrong metadata
**********************************************

The reconstruction relies on the ``startStep``/``endStep`` metadata of
the fields returned by the source. Some archives encode these
incorrectly; the ``patch:`` key applies named fixes to each field before
it is matched to the requested intervals:

-  ``set_start_step_to_zero`` — the archive encodes
   ``startStep == endStep`` but fields are accumulated from the start of
   the forecast;
-  ``reset_24h_accumulations`` — same, for archives whose accumulations
   reset every 24 h of lead time (see the reset example above).

``patch:`` describes what the fields' metadata *wrongly says*;
``accumulated:`` describes what the data *actually is*. They are
orthogonal and an archive may need both.

.. warning::

   If the data provided by the source does not match the archive
   description, the package checks the metadata of the source dataset
   and fails if the accumulation periods cannot be reconstructed.
   **If the metadata is incomplete or inconsistent, the package may
   produce incorrect results.**

******************************************************
 Controlling the fields regrouped within accumulation
******************************************************

It is possible to control the fields accumulated together through their
metadata. The ``group_by`` keyword allows to ignore some metadata when
deciding to group fields to accumulate them together. Ignored keys mean
that fields with different values will be accumulated together. Note
that ``date``, ``time`` and ``step`` must always be ignored.

.. literalinclude:: yaml/accumulate-groupby.yaml
   :language: yaml

*********************************************
 Forecast accumulations (trajectory recipes)
*********************************************

Inside a :ref:`trajectory recipe <layouts-trajectories>`,
``accumulate:`` produces per-step accumulation fields anchored on the
basetime imposed by the layout. An archive description key is **not
allowed** (the layout already dictates which trajectory serves each
window — declaring one is an error); the same ``accumulated:`` scheme
key is declared directly on the accumulate block, and is required:

.. code:: yaml

   base_dates: {start: 2021-01-01, end: 2021-01-03, frequency: 12h}
   steps:      {start: 6h, end: 30h, frequency: 3h}

   input:
     join:
       - pipe:
           - accumulate:
               period: 1h
               accumulated: from-zero
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

All three ``accumulated:`` values are supported, including
``from-zero-reset-every-<frequency>``. Recipe validation additionally
requires ``steps.start >= period`` — output windows must not straddle
the basetime.

.. note::

   For ECMWF forecasts, the forecasts at 00Z and 12Z are from the stream
   ``oper`` while the forecasts at 06Z and 18Z are from the stream
   ``scda``.
