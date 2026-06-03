.. _recipe:

########
 Recipe
########

A recipe is a YAML file that describes how to build a dataset. It is
composed of a list of sources and filters, and the operations to combine
them. Below is an example of a recipe. The order of the entries is not
important, but we recommend following the order of the example for
readability.

.. literalinclude:: syntax.yaml
   :language: yaml

Layout-specific recipe keys
===========================

The top-level dates block depends on ``output.layout``:

-  For ``layout: gridded`` (default) and ``layout: tabular`` — provide
   ``dates: {start, end, frequency, …}``. ``base_dates:`` and ``steps:``
   must be absent.
-  For ``layout: trajectories`` — provide ``base_dates: {...}`` and
   ``steps: {start, end, frequency}``. ``dates:`` must be absent. See
   :ref:`layouts-trajectories` for the full layout description.

The recipe loader validates these rules and raises a clear error if they
are mismatched.

Deprecated / removed output keys
================================

-  ``output.order_by`` is **deprecated**. The cube ordering is now
   hard-coded per layout (``["valid_datetime", "param_level", "number"]``
   for gridded, ``["traj_point", "param_level", "number"]`` for
   trajectories). Existing recipes that set it to the default value keep
   parsing with a ``DeprecationWarning``; any other value is rejected.
-  ``output.flatten_grid`` has been **removed**. Cube flattening is
   always on.
-  MARS request keys ``user_date`` and ``user_time`` are no longer
   accepted — use the wildcard shorthand ``date: "????-??-01"`` with an
   optional ``time:`` list instead. See :ref:`sources-mars`.
