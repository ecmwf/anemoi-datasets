.. _sources-from-trajectories:

###################
 from-trajectories
###################

The ``from-trajectories`` source lets a regular :ref:`gridded
<layouts-gridded>` recipe (driven by a ``dates:`` block) pull fields
from a **forecast** archive by picking, for each validity time, a
``(basetime, step)`` pair that produces it.

It is the inverse of the :ref:`trajectories layout
<layouts-trajectories>`: instead of materialising a 5-D
``(base_dates, variables, ensembles, steps, cells)`` array, it returns a
plain 4-D gridded dataset whose fields happen to come from forecast
runs.

Recipe shape
============

.. code:: yaml

   dates: {start: 2023-01-01, end: 2023-01-10, frequency: 6h}

   input:
     from-trajectories:
       bases: "????-??-?? 00:00:00"     # optional fnmatch pattern on basetime
       steps: 6/to/24/by/6              # optional step spec
       source:
         mars:
           type: fc
           class: od
           expver: "0001"
           param: [t, q]
           levtype: pl
           level: [500, 850]

   output:
     layout: gridded

Parameters
==========

-  **bases** (optional) — an fnmatch-style wildcard pattern matched
   against the basetime formatted as ``"%Y-%m-%d %H:%M:%S"``. ``?``
   matches any single character and ``*`` matches any sequence. Omit to
   accept any basetime.
-  **steps** (optional) — a MARS-style step specification: an integer,
   a list, or a string such as ``"6/to/24/by/6"``. Omit to default to
   step 0 (analysis-like).
-  **source** (required) — a single-key dict describing the inner
   source. Today this must be a source that supports the forecast
   dispatch (``mars:``).

Resolution strategy
===================

For each validity time requested by the recipe, candidate basetimes are
computed as ``valid_time − step`` for every configured step, smallest
step first. The first candidate that matches the ``bases`` pattern is
kept. If none matches, a clear ``ValueError`` is raised at build time —
so a misconfigured recipe fails loudly rather than producing a silently
empty dataset.

The inner source is then invoked once with the resulting
``(valid_time, basetime)`` pairs, via the same forecast-dispatch path
used inside a trajectory recipe.
