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

.. literalinclude:: code/matching0_.py
   :language: python

or more than one attribute:

.. literalinclude:: code/matching1_.py
   :language: python

You can also use `dates` as a shortcut for the above. This is equivalent
to:

.. literalinclude:: code/matching2_.py
   :language: python

To use the common set of variables, use:

.. literalinclude:: code/matching3_.py
   :language: python

To match all the attributes:

.. literalinclude:: code/matching4_.py
   :language: python
