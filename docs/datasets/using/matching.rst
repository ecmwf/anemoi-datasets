.. _using-matching:

#####################
 Matching attributes
#####################

When :ref:`combining datasets <combining-datasets>` with operations like
:ref:`concat`, :ref:`join`, :ref:`ensembles` or :ref:`grids`, some of
the attributes of the input datasets must match, such as the list of
variables for `concat` or the `dates` and `frequency` for `join`.

You can let the package automatically adjust the attributes of the input
datasets using the `adjust` keyword, to adjust one of the attributes:

.. code:: python

   ds = open_dataset(
       join=[dataset1, dataset2],
       adjust="frequency",
   )

or more than one attribute:

.. code:: python

   ds = open_dataset(
       join=[dataset1, dataset2],
       adjust=["start", "end", "frequency"],
   )

You can also use `dates` as a shortcut for the above. This is equivalent
to:

.. code:: python

   ds = open_dataset(join=[dataset1, dataset2], adjust="dates")

To use the common set of variables, use:

.. code:: python

   ds = open_dataset(concat=[dataset1, dataset2], adjust="variables")

To match all the attributes:

.. code:: python

   ds = open_dataset(
       cutout=[dataset1, dataset2],
       adjust="all",
   )
